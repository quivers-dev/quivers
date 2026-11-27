"""Independent closed forms for fixed words with productive self-loops."""

from fractions import Fraction
from math import log, nextafter

import pytest
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import (
    COUNTING,
    LOG_PROB,
    MARKOV,
    REAL,
    DivergentClosureError,
    UndecidedClosureError,
)
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
from tests.evidence.oracles.closed_forms import Divergent, geometric


def _reference(tied, zero, other, *, emits):
    """Exact mass and derivatives for two parallel tied loops and one zero slot."""
    first = geometric(2 * tied + zero)
    if emits:
        # A half-weight transition reaches a second loop and a tied final mark.
        second = geometric(other)
        entry = tied**2 / 2
        return entry * first * second, {
            "tied": tied * first * second + 2 * entry * first**2 * second,
            "zero": entry * first**2 * second,
            "other": entry * first * second**2,
        }
    return tied * first, {
        "tied": first + 2 * tied * first**2,
        "zero": tied * first**2,
        "other": Fraction(0),
    }


def test_geometric_oracle_rejects_dropped_occurrences_and_finite_unrolling():
    tied = Fraction(1, 8)
    zero = Fraction(0)
    expected, derivative = _reference(tied, zero, Fraction(1, 4), emits=False)
    assert expected == Fraction(1, 6)
    assert derivative == {"tied": Fraction(16, 9), "zero": Fraction(2, 9), "other": 0}
    accuracy = Accuracy(domain="linear", operations=32)
    wrong = (
        tied * geometric(tied + zero),
        tied * sum((2 * tied + zero) ** length for length in range(17)),
        tied * geometric(tied) ** 2,
    )
    for mass in wrong:
        with pytest.raises(Mismatch):
            compare_number(float(mass), expected, accuracy, "parallel loop mass")
    with pytest.raises(Divergent):
        _reference(Fraction(1, 2), zero, Fraction(0), emits=False)


_TAPE = StringSet(alphabet=FinSet(name="symbol", cardinality=1))


def _two_components():
    loop = Arc(0, 0, (None,), 0)
    return RelationTopology(
        (_TAPE,),
        1,
        2,
        (loop, loop, Arc(0, 0, (None,), 1), Arc(0, 1, (0,), 4), Arc(1, 1, (None,), 2)),
        (Boundary(0, 0),),
        (Boundary(0, 3), Boundary(1, 0)),
        ("tied", "zero", "other", "unit", "step"),
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_live_loop_closures_and_derivatives_match_exact_geometric_products(algebra):
    is_log = algebra is LOG_PROB
    slots = nn.Parameter(torch.zeros(5, dtype=torch.float64))

    # The topology is compiled without sampling weight values. One
    # original parameter tensor supplies every slot at evaluation.
    relation = RationalRelation(_two_components(), slots, algebra=algebra)
    epsilon = relation.query(((),), max_closure_size=3)
    emitted = relation.query(((0,),), max_closure_size=3)
    assert epsilon.work == QueryWork(1, 3, 1, 3, (3,))
    assert emitted.work == QueryWork(2, 5, 2, 5, (3, 1))
    assert list(relation.module().parameters())[0] is slots

    accuracy = Accuracy(domain="linear", operations=220)
    for tied, zero, other in (
        (Fraction(1, 8), Fraction(0), Fraction(1, 4)),
        (Fraction(3, 16), Fraction(1, 8), Fraction(1, 3)),
    ):
        masses = (tied, zero, other, Fraction(1), Fraction(1, 2))
        values = torch.tensor(tuple(map(float, masses)), dtype=slots.dtype)
        with torch.no_grad():
            slots.copy_(values.log() if is_log else values)
        for emits, query in ((False, epsilon), (True, emitted)):
            value = query.weight()
            (gradient,) = torch.autograd.grad(value, slots)
            mass, derivatives = _reference(tied, zero, other, emits=emits)
            if is_log:
                derivatives = {
                    name: derivative * input_mass / mass
                    for (name, derivative), input_mass in zip(
                        derivatives.items(), masses[:3], strict=True
                    )
                }
            compare_number(
                (value.exp() if is_log else value).item(),
                mass,
                accuracy,
                "closed path mass",
            )
            compare_gradients(
                dict(zip(derivatives, gradient[:3], strict=True)),
                derivatives,
                accuracy,
                "closed path derivatives",
            )


def _self_cycles(*weight_names):
    return RelationTopology(
        (_TAPE,),
        1,
        1,
        tuple(Arc(0, 0, (None,), i) for i in range(len(weight_names))),
        (Boundary(0, len(weight_names)),),
        (Boundary(0, len(weight_names)),),
        (*weight_names, "unit"),
    )


def test_runtime_oracle_rejects_deduplicated_parallel_loop_slots():
    relation = RationalRelation(
        _two_components(),
        torch.tensor([0.125, 0.0, 0.25, 1.0, 0.5], dtype=torch.float64),
        algebra=MARKOV,
    )
    query = relation.query(((),), max_closure_size=3)
    expected, _ = _reference(Fraction(1, 8), Fraction(0), Fraction(1, 4), emits=False)

    def check():
        compare_number(
            query.weight().item(),
            expected,
            Accuracy(domain="linear", operations=32),
            "parallel loop slots",
        )

    check()
    coalesced = Mutation(
        name="deduplicated-closure-occurrences",
        defect_class="M14",
        provenance="treat every occurrence of the same slot as one loop",
        target="quivers.core.relations.queries:_close_loops",
        rewrites=(
            Rewrite(
                old="auxiliary = len(slots) - 1",
                new="slots = tuple(dict.fromkeys(slots))\n    auxiliary = len(slots) - 1",
            ),
        ),
        rejected_by=("parallel-loop-slots",),
        min_effect=0.02,
    )
    with installed(coalesced), pytest.raises(Mismatch) as refused:
        check()
    assert refused.value.deviation >= coalesced.min_effect


def test_query_refuses_unsupported_closures_and_checks_block_order():
    topology = _two_components()
    weights = torch.tensor([0.125, 0.0, 0.25, 1.0, 0.5], dtype=torch.float64)
    relation = RationalRelation(topology, weights, algebra=MARKOV)
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon cycle"):
        relation.weight(((),))
    with pytest.raises(RelationLimitError, match="max_closure_size"):
        relation.query(((),), max_closure_size=2)
    with pytest.raises(ValueError, match="max_closure_size"):
        relation.query(((),), max_closure_size=True)
    with pytest.raises(ProductiveEpsilonCycleError, match="established semiring star"):
        RationalRelation(topology, weights, algebra=REAL).query(
            ((),), max_closure_size=3
        ).weight()

    mutual = RelationTopology(
        (_TAPE,),
        1,
        2,
        (Arc(0, 1, (None,), 0), Arc(1, 0, (None,), 0)),
        (Boundary(0, 1),),
        (Boundary(0, 1),),
        ("change", "unit"),
    )
    with pytest.raises(ProductiveEpsilonCycleError, match="singleton"):
        RationalRelation(mutual, torch.tensor([0.25, 1.0]), algebra=MARKOV).query(
            ((),), max_closure_size=2
        )


def test_closure_rechecks_divergence_after_a_weight_update():
    weights = nn.Parameter(torch.tensor([0.25, 0.25, 1.0], dtype=torch.float64))
    relation = RationalRelation(_self_cycles("left", "right"), weights, algebra=MARKOV)
    query = relation.query(((),), max_closure_size=2)
    assert query.weight().item() == geometric(Fraction(1, 2))
    with torch.no_grad():
        weights[:2].fill_(0.5)
    with pytest.raises(DivergentClosureError):
        query.weight()
    with torch.no_grad():
        weights.fill_(0.0)
        weights[2] = 1.0
    assert query.weight().item() == 1
    assert query.work == QueryWork(1, 2, 1, 2, (2,))


def test_boundary_rounding_refuses_instead_of_returning_an_unusable_weight():
    below = nextafter(1.0, 0.0)
    small = 2**-54
    exact_ratio = Fraction(below) + Fraction(small)
    assert exact_ratio < 1 and below + small == 1.0
    topology = _self_cycles("large", "small")
    weights = torch.tensor([below, small, 1.0], dtype=torch.float64)
    query = RationalRelation(topology, weights, algebra=MARKOV).query(
        ((),), max_closure_size=2
    )
    try:
        value = query.weight().item()
    except UndecidedClosureError:
        pass
    else:
        compare_number(
            value,
            geometric(exact_ratio),
            Accuracy(domain="linear", operations=32),
            "near-boundary closure",
        )

    log_weights = torch.tensor([-log(2), -log(2), 0.0], dtype=torch.float64)
    with pytest.raises(UndecidedClosureError, match="rounding"):
        RationalRelation(topology, log_weights, algebra=LOG_PROB).query(
            ((),), max_closure_size=2
        ).weight()


def test_counting_keeps_structural_zero_loops_and_refuses_positive_ones():
    topology = _self_cycles("loop")
    integers = RationalRelation(topology, torch.tensor([0, 1]), algebra=COUNTING)
    with pytest.raises(TypeError, match="requires floating-point weights"):
        integers.query(((),), max_closure_size=1).weight()
    values = torch.tensor([0.0, 1.0], dtype=torch.float64)
    query = RationalRelation(topology, values, algebra=COUNTING).query(
        ((),), max_closure_size=1
    )
    assert query.weight().item() == 1
    assert query.work.closure_sizes == (1,)
    values[0] = 1
    with pytest.raises(DivergentClosureError):
        query.weight()


def test_closure_keeps_the_weight_of_a_very_unlikely_log_path():
    slots = nn.Parameter(torch.tensor([log(0.25), -500.0], dtype=torch.float64))
    topology = _self_cycles("loop")
    query = RationalRelation(topology, slots, algebra=LOG_PROB).query(
        ((),), max_closure_size=1
    )
    mass = geometric(Fraction(1, 4))
    score = query.weight()
    compare_number(
        score.item(),
        -1000 + log(mass),
        Accuracy(domain="log", operations=72, magnitude=1000),
        "tiny accepting weight with a loop",
    )
    (derivative,) = torch.autograd.grad(score, slots)
    compare_gradients(
        dict(zip(("loop", "boundary"), derivative, strict=True)),
        {"loop": Fraction(1, 3), "boundary": Fraction(2)},
        Accuracy(domain="linear", operations=72),
        "tiny path log derivatives",
    )


def test_nonproductive_cycle_consumes_no_closure_budget():
    topology = RelationTopology(
        (_TAPE,),
        1,
        2,
        (Arc(0, 1, (None,), 0), Arc(1, 1, (None,), 0)),
        (Boundary(0, 0),),
        (Boundary(0, 0),),
        ("unit",),
    )
    query = RationalRelation(topology, torch.ones(1), algebra=MARKOV).query(
        ((),), max_closure_size=0
    )
    assert query.work == QueryWork(2, 2, 1, 0)
    assert query.weight().item() == 1
