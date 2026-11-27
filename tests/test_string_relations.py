"""Exact finite observations of a sparse rational relation."""

from fractions import Fraction
from itertools import product
from math import prod

import pytest
import torch
from torch import nn

from quivers import FiniteMorphism, FinSet, Morphism, Program, StringSet, observed
from quivers.continuous.morphisms import DiscreteAsContinuous
from quivers.core.algebras import COUNTING, LOG_PROB, REAL
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    QueryWork,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
)
from tests.evidence.compare import Accuracy, Mismatch, compare_gradients, compare_number
from tests.evidence.mutations import Mutation, Rewrite, installed
from tests.evidence.oracles import paths


class LegacyTensorMorphism(Morphism):
    def __init__(self, obj):
        super().__init__(obj, obj, REAL)
        self.weight = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.weight.weight.fill_(2.0)
        self.reads = 0

    @property
    def tensor(self):
        self.reads += 1
        return self.weight.weight.reshape(self.tensor_shape)

    def module(self):
        return self.weight


def test_classifies_existing_custom_tensor_morphism_without_reading_tensor():
    obj = FinSet(name="one", cardinality=1)
    morphism = LegacyTensorMorphism(obj)
    assert isinstance(morphism, FiniteMorphism)
    composed = morphism >> morphism
    assert morphism.reads == 0
    torch.testing.assert_close(composed.tensor, torch.tensor([[4.0]]))
    assert morphism.reads == 2
    composed.tensor.sum().backward()
    torch.testing.assert_close(morphism.weight.weight.grad, torch.tensor([[4.0]]))


def test_string_carrier_requires_a_checked_bound():
    binary = StringSet(alphabet=FinSet(name="bits", cardinality=2))
    with pytest.raises(TypeError, match="unbounded"):
        _ = binary.size
    with pytest.raises(TypeError, match="finite tensor shape"):
        _ = binary.shape
    with pytest.raises(ValueError, match="max_size"):
        binary.bounded(1 << 64, max_size=100)
    bounded = binary.bounded(3, max_size=15)
    oracle_words = [
        word for length in range(4) for word in product(range(2), repeat=length)
    ]
    assert bounded.size == 15
    assert [bounded.decode(i) for i in range(15)] == oracle_words
    assert [bounded.encode(word) for word in oracle_words] == list(range(15))


def test_unary_bounded_carrier_does_not_build_a_coproduct_to_find_its_shape(
    monkeypatch,
):
    unary = StringSet(alphabet=FinSet(name="unary", cardinality=1))

    def fail(*args):
        pytest.fail("an arithmetic size query built the coproduct")

    monkeypatch.setattr("quivers.core.objects._build_free_monoid_components", fail)
    bounded = unary.bounded(1 << 20, max_size=(1 << 20) + 1)
    assert bounded.shape == ((1 << 20) + 1,)
    assert bounded.encode((0,) * 50) == 50
    assert bounded.decode(50) == (0,) * 50


def _observed_machine():
    return paths.Machine(
        initial=(paths.Mark(state=0, weight="start"),),
        final=(paths.Mark(state=0, weight="stop"), paths.Mark(state=3, weight="stop")),
        arcs=(
            paths.Arc(
                name="first-copy", source=0, target=1, labels=("a", None), weight="tied"
            ),
            paths.Arc(
                name="second-copy",
                source=0,
                target=1,
                labels=("a", None),
                weight="tied",
            ),
            paths.Arc(
                name="then-insert",
                source=1,
                target=3,
                labels=(None, "b"),
                weight="tied",
            ),
            paths.Arc(
                name="silent", source=0, target=2, labels=(None, None), weight="start"
            ),
            paths.Arc(
                name="substitution",
                source=2,
                target=3,
                labels=("a", "b"),
                weight="other",
            ),
        ),
    )


def _topology():
    tape = StringSet(alphabet=FinSet(name="letters", cardinality=2))
    return RelationTopology(
        tapes=(tape, tape),
        input_tapes=1,
        n_states=4,
        arcs=(
            Arc(0, 1, (0, None), 0),
            Arc(0, 1, (0, None), 0),
            Arc(1, 3, (None, 1), 0),
            Arc(0, 2, (None, None), 1),
            Arc(2, 3, (0, 1), 3),
        ),
        initial=(Boundary(0, 1),),
        final=(Boundary(0, 2), Boundary(3, 2)),
        weight_names=("tied", "start", "stop", "other"),
    )


@pytest.mark.parametrize("algebra", [REAL, LOG_PROB, COUNTING])
def test_bounded_tensor_agrees_with_independent_path_enumeration(algebra):
    numbers = (2, 3, 5, 7)
    weight_of = {
        name: Fraction(value)
        for name, value in zip(("tied", "start", "stop", "other"), numbers, strict=True)
    }
    relation_table = paths.relation(_observed_machine(), weight_of, tapes=2)
    values = torch.tensor(numbers, dtype=torch.float64)
    if algebra is LOG_PROB:
        values = values.log()
    machine = RationalRelation(_topology(), values, algebra=algebra)

    outputs = [(), ("a",), ("b",)]
    expected = torch.tensor(
        [
            [float(relation_table.get((x, y), Fraction(0))) for y in outputs]
            for x in outputs
        ],
        dtype=torch.float64,
    )
    restricted = machine.restrict((1, 1), max_entries=9)
    assert isinstance(restricted, FiniteMorphism)
    actual = restricted.tensor
    if algebra is LOG_PROB:
        actual = actual.exp()
    torch.testing.assert_close(actual, expected)
    pair = machine.weight(((0,), (1,)))
    if algebra is LOG_PROB:
        pair = pair.exp()
    torch.testing.assert_close(pair, expected[1, 2])
    assert not hasattr(machine, "tensor")
    assert not isinstance(machine, FiniteMorphism)
    with pytest.raises(ValueError, match="max_entries"):
        machine.restrict((1 << 40, 1), max_entries=9)
    with pytest.raises(ValueError, match="max_entries"):
        machine.restrict((1, 1), max_entries=8)
    with pytest.raises(TypeError, match="explicit finite restriction"):
        machine @ observed(restricted.domain, restricted.codomain, expected, REAL)


def _check_live_weight_slots():
    class Scores(nn.Module):
        def __init__(self):
            super().__init__()
            self.tied = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        def forward(self):
            return torch.stack(
                (self.tied, self.tied.new_tensor(3), self.tied.new_tensor(5), self.tied)
            )

    scores = Scores()
    machine = RationalRelation(_topology(), scores, algebra=REAL)
    restricted = machine.restrict((1, 1), max_entries=9)
    params = list(restricted.module().parameters())
    assert len(params) == 1 and params[0] is scores.tied
    optimizer = torch.optim.SGD(params, lr=0.01, maximize=True)
    accepted = [
        path
        for path in paths.accepting_paths(_observed_machine())
        if path.strings() == (("a",), ("b",))
    ]
    # Each accepting path has at most four factors. The value and
    # derivative sums here have nonnegative terms.
    accuracy = Accuracy(domain="linear", operations=32)
    for _ in range(2):
        current = Fraction.from_float(scores.tied.item())
        weights = {
            "tied": current,
            "start": Fraction(3),
            "stop": Fraction(5),
            "other": current,
        }
        expected = paths.total(accepted, weights, Fraction(0), Fraction(1))
        derivative = Fraction(0)
        for path in accepted:
            factors = path.factors()
            for i, name in enumerate(factors):
                if name in {"tied", "other"}:
                    derivative += prod(
                        weights[other] for j, other in enumerate(factors) if i != j
                    )

        optimizer.zero_grad()
        actual = restricted.tensor[1, 2]
        compare_number(actual.item(), expected, accuracy, "path weight")
        actual.backward()
        compare_gradients(
            {"tied": scores.tied.grad}, {"tied": derivative}, accuracy, "path weight"
        )
        optimizer.step()


def test_live_weight_slots_preserve_shared_parameter_through_optimizer_updates():
    _check_live_weight_slots()


@pytest.mark.parametrize(
    "mutation",
    (
        Mutation(
            name="string-relation-occurrence-detached",
            defect_class="M7",
            provenance="use a detached value for the substitution occurrence of a shared parameter",
            target="quivers.core.relations.queries:_PathPlan.evaluate",
            rewrites=(
                Rewrite(
                    old="value = algebra.tensor_op(prefix, weights[slot])",
                    new=(
                        "weight = weights[slot].detach() if arc_id == 4 else weights[slot]"
                        "\n            value = algebra.tensor_op(prefix, weight)"
                    ),
                ),
            ),
            rejected_by=("live-weight-slots",),
            min_effect=1.0,
        ),
        Mutation(
            name="string-relation-weights-cached-across-updates",
            defect_class="M24",
            provenance="cache a vector of numerical weights along with the structural restriction",
            target="quivers.core.relations:RationalRelation._weights",
            rewrites=(
                Rewrite(
                    old="values = self._weight_source()",
                    new=(
                        'if not hasattr(self, "_cached_weights"):\n'
                        "        self._cached_weights = self._weight_source()\n"
                        "    values = self._cached_weights"
                    ),
                ),
            ),
            rejected_by=("live-weight-slots",),
            min_effect=1.0,
        ),
    ),
    ids=lambda mutation: mutation.name,
)
def test_live_weight_check_rejects_parameter_mutations(mutation):
    _check_live_weight_slots()
    with installed(mutation):
        with pytest.raises(Mismatch) as caught:
            _check_live_weight_slots()
    assert caught.value.deviation >= mutation.min_effect


def test_program_requires_explicit_restriction_but_accepts_legacy_finite_morphisms():
    obj = FinSet(name="one", cardinality=1)
    legacy = LegacyTensorMorphism(obj)
    torch.testing.assert_close(Program(legacy)(), torch.tensor([[2.0]]))

    relation = RationalRelation(_topology(), torch.tensor([2.0, 3.0, 5.0, 7.0]))
    with pytest.raises(TypeError, match="explicit finite restriction"):
        Program(relation)
    with pytest.raises(TypeError, match="explicit finite restriction"):
        DiscreteAsContinuous(relation)
    bounded = relation.restrict((0, 0), max_entries=1)
    torch.testing.assert_close(Program(bounded)(), torch.tensor([[15.0]]))


def test_fixed_word_plan_matches_an_emitting_cycle_and_refuses_a_live_silent_cycle():
    tape = StringSet(alphabet=FinSet(name="letters", cardinality=2))
    topology = RelationTopology(
        tapes=(tape,),
        input_tapes=1,
        n_states=4,
        arcs=(
            Arc(0, 1, (0,), 0),
            Arc(1, 1, (0,), 0),
            Arc(0, 2, (None,), 1),
            Arc(2, 2, (None,), 2),
            Arc(2, 3, (1,), 0),
        ),
        initial=(Boundary(0, 1),),
        final=(Boundary(1, 1), Boundary(3, 1)),
        weight_names=("emit", "unit", "zero"),
    )
    machine = RationalRelation(topology, torch.tensor([2.0, 1.0, 0.0]))
    assert machine.weight(((0, 0, 0),)).item() == 8.0
    assert machine.weight(((),)).item() == 0.0
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon cycle"):
        machine.weight(((1,),))


def test_label_index_skips_unmatched_arcs_and_enforces_exact_construction_limits():
    size = 2500
    tape = StringSet(alphabet=FinSet(name="letters", cardinality=size))
    topology = RelationTopology(
        tapes=(tape,),
        input_tapes=1,
        n_states=1 << 32,
        arcs=tuple(Arc(0, 1, (index,), 0) for index in range(size)),
        initial=(Boundary(0, 0),),
        final=(Boundary(1, 0),),
        weight_names=("unit",),
    )
    machine = RationalRelation(topology, torch.tensor([1.0]))
    observed = ((size - 1,),)
    query = machine.query(observed, max_states=2, max_work=1)
    assert query.work == QueryWork(2, 1, 2, 1)
    assert query.weight().item() == 1.0
    with pytest.raises(RelationLimitError, match="max_states"):
        machine.query(observed, max_states=1)
    with pytest.raises(RelationLimitError, match="max_work"):
        machine.query(observed, max_work=0)
    bounded = machine.restrict((1,), max_entries=size + 1, max_states=2)
    with pytest.raises(RelationLimitError, match="max_states"):
        _ = bounded.tensor


def test_multi_tape_restriction_preserves_order_and_tape_specific_symbols():
    unary = StringSet(alphabet=FinSet(name="source", cardinality=1))
    binary = StringSet(alphabet=FinSet(name="target", cardinality=2))
    topology = RelationTopology(
        tapes=(unary, binary, binary),
        input_tapes=2,
        n_states=2,
        arcs=(Arc(0, 1, (0, 1, 0), 0),),
        initial=(Boundary(0, 0),),
        final=(Boundary(1, 0),),
        weight_names=("unit",),
    )
    machine = RationalRelation(topology, torch.tensor([1.0]))
    assert machine.domain == unary * binary
    assert machine.codomain == binary
    assert machine.weight(((0,), (1,), (0,))).item() == 1.0
    assert machine.weight(((0,), (0,), (1,))).item() == 0.0
    with pytest.raises(ValueError, match="invalid symbol on tape 0"):
        machine.weight(((1,), (1,), (0,)))
    restricted = machine.restrict((1, 1, 1), max_entries=18)
    table = restricted.tensor
    assert table.shape == (2, 3, 3)
    assert table.nonzero().tolist() == [[1, 2, 1]]


def test_one_word_tuple_never_uses_a_finite_tensor_adapter(monkeypatch):
    relation = RationalRelation(_topology(), torch.tensor([2.0, 3.0, 5.0, 7.0]))

    def reject_adapter(*args, **kwargs):
        raise Mismatch(
            "work", float("inf"), "fixed-word query constructed a tensor adapter"
        )

    monkeypatch.setattr(RationalRelation, "restrict", reject_adapter)
    assert relation.weight(((0,), (1,))).item() == 435.0
    mutation = Mutation(
        name="fixed-word-query-uses-dense-adapter",
        defect_class="M31",
        provenance="materialize a bounded relation before handling an individual observation",
        target="quivers.core.relations:RationalRelation.weight",
        rewrites=(
            Rewrite(
                old="return self.query(words).weight()",
                new="return self.restrict(tuple(map(len, words)), max_entries=9).tensor.ravel()[0]",
            ),
        ),
        rejected_by=("one-word-tuple-no-dense-adapter",),
        min_effect=1.0,
    )
    with installed(mutation):
        with pytest.raises(Mismatch, match="constructed a tensor adapter") as caught:
            relation.weight(((0,), (1,)))
    assert caught.value.deviation >= mutation.min_effect
