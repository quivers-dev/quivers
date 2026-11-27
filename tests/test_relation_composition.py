"""Independent path-pair oracles for sparse fixed-word composition."""

from fractions import Fraction
from itertools import product
from math import log, prod

import pytest
import torch
from torch import nn

from quivers import FinSet, Program, StringSet
from quivers.core.algebras import COUNTING, LOG_PROB, REAL, RealAlgebra
from quivers.core.relations import (
    Arc,
    Boundary,
    ComposedRelation,
    ProductiveEpsilonCycleError,
    QueryWork,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
)
from quivers.qiec import ParameterId, RationalRelationSpec, dumps, loads
from tests.evidence.compare import Accuracy, Mismatch, compare_gradients, compare_number
from tests.evidence.mutations import Mutation, Rewrite, installed
from tests.evidence.oracles import paths


_INPUT = StringSet(alphabet=FinSet(name="input", cardinality=2))
_SHARED = StringSet(alphabet=FinSet(name="shared", cardinality=2))
_OUTPUT = StringSet(alphabet=FinSet(name="output", cardinality=2))
_IDS = {"a": 0, "b": 1, "x": 0, "q": 1, "z": 0, "r": 1, None: None}


def _words(pair):
    return tuple(tuple(_IDS[symbol] for symbol in word) for word in pair)


def _topology(machine, tapes):
    names = tuple(
        dict.fromkeys(
            item.weight for item in (*machine.arcs, *machine.initial, *machine.final)
        )
    )
    slots = {name: i for i, name in enumerate(names)}
    return RelationTopology(
        tapes=tapes,
        input_tapes=1,
        n_states=1
        + max(
            max(mark.state for mark in (*machine.initial, *machine.final)),
            max((max(arc.source, arc.target) for arc in machine.arcs), default=0),
        ),
        arcs=tuple(
            Arc(
                arc.source,
                arc.target,
                tuple(_IDS[label] for label in arc.labels),
                slots[arc.weight],
            )
            for arc in machine.arcs
        ),
        initial=tuple(
            Boundary(mark.state, slots[mark.weight]) for mark in machine.initial
        ),
        final=tuple(Boundary(mark.state, slots[mark.weight]) for mark in machine.final),
        weight_names=names,
    )


def _chain(labels, *, left):
    outer = "x" if left else "z"
    arcs = []
    for i, shared in enumerate(labels):
        emitted = None if i % 2 else outer
        arcs.append(
            paths.Arc(
                name=str(i),
                source=i,
                target=i + 1,
                labels=(emitted, shared) if left else (shared, emitted),
                weight="unit",
            )
        )
    return paths.Machine(
        arcs=tuple(arcs),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=(paths.Mark(state=len(labels), weight="unit"),),
    )


def test_composition_matches_exhaustive_short_abstract_words_and_epsilon_runs():
    labels = (None, "a", "b")
    sequences = [word for size in range(4) for word in product(labels, repeat=size)]
    observed_pairs = list(product(sequences, repeat=2))
    # Residual left steps before the match and residual right steps after it
    # require the filter to reset at the shared symbol.
    observed_pairs.append(((None, None, "a", None), (None, "a", None, None)))
    unit = {"unit": Fraction(1)}
    for l_labels, r_labels in observed_pairs:
        left = _chain(l_labels, left=True)
        right = _chain(r_labels, left=False)
        l_path = paths.accepting_paths(left)[0]
        r_path = paths.accepting_paths(right)[0]
        l_word, _ = l_path.strings() or ((), ())
        _, r_word = r_path.strings() or ((), ())
        left_relation = RationalRelation(
            _topology(left, (_INPUT, _SHARED)), torch.ones(1), algebra=COUNTING
        )
        right_relation = RationalRelation(
            _topology(right, (_SHARED, _OUTPUT)), torch.ones(1), algebra=COUNTING
        )
        composition = ComposedRelation(left_relation, right_relation)
        expected = paths.compose_relations(
            paths.relation(left, unit, tapes=2), paths.relation(right, unit, tapes=2)
        )
        assert composition.weight(_words((l_word, r_word))).item() == expected.get(
            (l_word, r_word), 0
        ), (l_labels, r_labels)


def test_nonidempotent_composition_oracle_rejects_a_removed_epsilon_filter():
    left = _chain((None,), left=True)
    right = _chain((None,), left=False)
    unit = {"unit": Fraction(1)}
    expected = paths.compose_relations(
        paths.relation(left, unit, tapes=2), paths.relation(right, unit, tapes=2)
    )[(("x",), ("z",))]
    composition = ComposedRelation(
        RationalRelation(
            _topology(left, (_INPUT, _SHARED)), torch.ones(1), algebra=COUNTING
        ),
        RationalRelation(
            _topology(right, (_SHARED, _OUTPUT)), torch.ones(1), algebra=COUNTING
        ),
    )

    def check():
        compare_number(
            composition.weight(((0,), (0,))).item(),
            expected,
            Accuracy(domain="linear", operations=8),
            "multiplicity of a compatible epsilon pair",
        )

    check()
    removed = Mutation(
        name="unfiltered-sparse-composition",
        defect_class="M1",
        provenance="allow both unilateral orders as well as the paired epsilon move",
        target="quivers.core.relations.calculus:_moves",
        rewrites=(
            Rewrite(old="if state is not _Filter.RIGHT:", new="if True:"),
            Rewrite(old="if state is not _Filter.LEFT:", new="if True:"),
            Rewrite(
                old="if label is EPSILON and state is not _Filter.MATCH:",
                new="if False:",
            ),
        ),
        rejected_by=("composed-epsilon-multiplicity",),
        min_effect=2.0,
    )
    with installed(removed), pytest.raises(Mismatch) as caught:
        check()
    assert caught.value.deviation == 2


def _weighted_machines():
    left = paths.Machine(
        arcs=(
            paths.Arc(
                name="copy1", source=0, target=1, labels=("x", None), weight="tied"
            ),
            paths.Arc(
                name="copy2", source=0, target=1, labels=("x", None), weight="tied"
            ),
            paths.Arc(
                name="match", source=1, target=2, labels=(None, "a"), weight="tied"
            ),
            paths.Arc(
                name="zero", source=2, target=3, labels=("q", None), weight="zero"
            ),
        ),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=tuple(paths.Mark(state=i, weight="unit") for i in (0, 2, 3)),
    )
    right = paths.Machine(
        arcs=(
            paths.Arc(
                name="first", source=0, target=1, labels=(None, "z"), weight="tied"
            ),
            paths.Arc(
                name="match", source=1, target=2, labels=("a", None), weight="unit"
            ),
            paths.Arc(
                name="tail", source=2, target=3, labels=(None, "r"), weight="tail"
            ),
        ),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=tuple(paths.Mark(state=i, weight="unit") for i in (0, 2, 3)),
    )
    return left, right


@pytest.mark.parametrize("algebra", [REAL, LOG_PROB])
def test_parallel_paths_and_original_parameter_derivatives_survive_composition(algebra):
    left, right = _weighted_machines()
    ids = {
        name: ParameterId.derive("composition", name)
        for name in ("tied", "tail", "zero", "unit")
    }
    names = ("tied", "zero", "tail")
    parameters = {
        name: nn.Parameter(torch.empty((), dtype=torch.float64)) for name in names
    }
    is_log = algebra is LOG_PROB
    unit = torch.tensor(0.0 if is_log else 1.0, dtype=torch.float64)
    bindings = {ids[name]: parameters[name] for name in names} | {ids["unit"]: unit}

    def bind(machine, tapes):
        topology = _topology(machine, tapes)
        spec = RationalRelationSpec.of(
            topology, algebra, tuple(ids[name] for name in topology.weight_names)
        )
        restored = loads(dumps(spec))
        assert isinstance(restored, RationalRelationSpec)
        return restored.bind(bindings)

    pairs: dict[paths.Strings, list[list[str]]] = {}
    for l_path, r_path, _ in paths.compatible_path_pairs(left, right):
        x, _ = l_path.strings() or ((), ())
        _, z = r_path.strings() or ((), ())
        pairs.setdefault((x, z), []).append(l_path.factors() + r_path.factors())
    assert len(pairs[(("x",), ("z",))]) == 2

    queries = None
    for value in (Fraction(1, 2), Fraction(3, 4)):
        values = {
            "tied": value,
            "zero": Fraction(0),
            "tail": Fraction(2, 3),
            "unit": Fraction(1),
        }
        with torch.no_grad():
            for name, parameter in parameters.items():
                mass = values[name]
                parameter.fill_(
                    (log(mass) if mass else -float("inf")) if is_log else float(mass)
                )
        if queries is None:
            relation = ComposedRelation(
                bind(left, (_INPUT, _SHARED)), bind(right, (_SHARED, _OUTPUT))
            )
            assert {id(p) for p in relation.module().parameters()} == {
                id(p) for p in parameters.values()
            }
            assert relation.domain == _INPUT and relation.codomain == _OUTPUT
            assert not hasattr(relation, "tensor")
            with pytest.raises(TypeError, match="explicit finite restriction"):
                Program(relation)
            queries = {
                words: relation.query(_words(words))
                for words in (*pairs, (("q",), ("z",)))
            }

        table = paths.compose_relations(
            paths.relation(left, values, tapes=2),
            paths.relation(right, values, tapes=2),
        )
        accuracy = Accuracy(domain="linear", operations=144)
        for words, query in queries.items():
            expected = table.get(words, Fraction(0))
            gradients = dict.fromkeys(names, Fraction(0))
            for factors in pairs.get(words, ()):
                for i, name in enumerate(factors):
                    if name in gradients:
                        if is_log and expected:
                            gradients[name] += (
                                prod(values[k] for k in factors) / expected
                            )
                        elif not is_log:
                            gradients[name] += prod(
                                values[k] for j, k in enumerate(factors) if i != j
                            )
            score = query.weight()
            compare_number(
                (score.exp() if is_log else score).item(),
                expected,
                accuracy,
                "composed mass",
            )
            gradient = torch.autograd.grad(score, tuple(parameters.values()))
            compare_gradients(
                dict(zip(names, gradient)),
                gradients,
                accuracy,
                "composition derivative",
            )


def test_composition_work_counts_indexed_matches_and_honors_budgets():
    alphabet_size = 2500
    shared = StringSet(alphabet=FinSet(name="large", cardinality=alphabet_size))
    initial, final = (Boundary(0, 0),), (Boundary(1, 0),)
    left = RelationTopology(
        (_INPUT, shared),
        1,
        1 << 32,
        tuple(Arc(0, 1, (0, x), 0) for x in range(alphabet_size)),
        initial,
        final,
        ("unit",),
    )
    right = RelationTopology(
        (shared, _OUTPUT),
        1,
        1 << 32,
        (Arc(0, 1, (alphabet_size - 1, 0), 0),),
        initial,
        final,
        ("unit",),
    )
    relation = ComposedRelation(
        RationalRelation(left, torch.ones(1)), RationalRelation(right, torch.ones(1))
    )
    query = relation.query(((0,), (0,)), max_states=2, max_work=2)
    assert query.weight().item() == 1.0
    assert query.work == QueryWork(2, 1, 2, 1)
    assert query.label_probes == 1
    with pytest.raises(RelationLimitError, match="max_work"):
        relation.query(((0,), (0,)), max_work=1)
    with pytest.raises(RelationLimitError, match="max_states"):
        relation.query(((0,), (0,)), max_states=1)


def test_composition_refuses_mismatched_types_laws_and_productive_hidden_cycles():
    class Unestablished(RealAlgebra):
        def establishes(self, *laws):
            return False

    left = RelationTopology(
        (_INPUT, _SHARED),
        1,
        1,
        (Arc(0, 0, (None, 0), 1),),
        (Boundary(0, 0),),
        (Boundary(0, 0),),
        ("unit", "zero"),
    )
    right = RelationTopology(
        (_SHARED, _OUTPUT),
        1,
        1,
        (Arc(0, 0, (0, None), 1),),
        (Boundary(0, 0),),
        (Boundary(0, 0),),
        ("unit", "zero"),
    )
    w = torch.tensor([1.0, 0.0])
    left_relation = RationalRelation(left, w)
    right_relation = RationalRelation(right, w)
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon cycle"):
        ComposedRelation(left_relation, right_relation).weight(((), ()))
    with pytest.raises(TypeError, match="shared tape type"):
        ComposedRelation(right_relation, left_relation)
    with pytest.raises(TypeError, match="established commutative semiring"):
        ComposedRelation(left_relation, RationalRelation(right, w, algebra=COUNTING))
    algebra = Unestablished()
    with pytest.raises(TypeError, match="established commutative semiring"):
        ComposedRelation(
            RationalRelation(left, w, algebra=algebra),
            RationalRelation(right, w, algebra=algebra),
        )


def test_nonproductive_hidden_cycle_does_not_block_an_empty_path():
    initial = final = (Boundary(0, 0),)
    left = RelationTopology(
        (_INPUT, _SHARED),
        1,
        2,
        (Arc(0, 1, (None, 0), 0), Arc(1, 1, (None, 0), 0)),
        initial,
        final,
        ("unit",),
    )
    right = RelationTopology(
        (_SHARED, _OUTPUT),
        1,
        2,
        (Arc(0, 1, (0, None), 0), Arc(1, 1, (0, None), 0)),
        initial,
        final,
        ("unit",),
    )
    composition = ComposedRelation(
        RationalRelation(left, torch.ones(1)),
        RationalRelation(right, torch.ones(1)),
    )
    query = composition.query(((), ()))
    assert query.work == QueryWork(2, 2, 1, 0)
    assert query.weight().item() == 1.0
