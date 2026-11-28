"""Independent rational path and linear-system oracles for forward accuracy."""

from fractions import Fraction

import pytest
import torch

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB, MARKOV, DivergentClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationAccuracyError,
    RelationLimitError,
    RelationTopology,
    accuracy,
    certify_markov_path_weight,
    queries,
)

_TAPE = StringSet(alphabet=FinSet(name="Letters", cardinality=2))
_LIMITS = dict(
    max_states=32,
    max_work=64,
    max_exact_bits=4096,
    max_exact_work=4096,
    max_closure_cells=36,
    max_closure_size=6,
)
_TOL = Fraction(1, 10**12)


def _relation(
    states, arcs, initial, final, values, *, algebra=MARKOV, dtype=torch.float64
):
    topology = RelationTopology(
        (_TAPE,),
        0,
        states,
        tuple(arcs),
        tuple(initial),
        tuple(final),
        tuple(f"w{i}" for i in range(len(values))),
    )
    weights = torch.nn.Parameter(torch.tensor(values, dtype=dtype))
    return RationalRelation(topology, weights, algebra=algebra), weights


def _certify(relation, words=((0,),), **options):
    return certify_markov_path_weight(
        relation, words, **(_LIMITS | {"absolute_tolerance": _TOL} | options)
    )


def test_acyclic_bounds_are_exact_for_the_evaluated_binary_slots():
    relation, source = _relation(
        2, (Arc(0, 1, (0,), 0),) * 2, (Boundary(0, 0),), (Boundary(1, 0),), (0.1,)
    )
    value = Fraction(source.item())
    exact = 2 * value**3
    proof = _certify(relation, relative_tolerance=_TOL, max_closure_cells=0)
    scored = relation.weight(((0,),))
    assert torch.equal(proof.weight, scored)
    assert proof.lower == proof.upper == exact
    assert proof.absolute_error == abs(Fraction(proof.weight.item()) - exact) > 0
    assert proof.relative_error == proof.absolute_error / exact
    (gradient,) = torch.autograd.grad(proof.weight, source)
    assert gradient.item() == pytest.approx(float(6 * value**2))
    with pytest.raises(RelationAccuracyError, match="requested absolute path error"):
        _certify(relation, absolute_tolerance=0)
    with pytest.raises(RelationAccuracyError, match="requested relative path error"):
        _certify(relation, absolute_tolerance=None, relative_tolerance=0)


def _two_state():
    return _relation(
        3,
        (
            Arc(0, 0, (None,), 1),
            Arc(0, 1, (None,), 0),
            Arc(0, 1, (None,), 0),
            Arc(1, 0, (None,), 2),
            Arc(1, 1, (None,), 1),
            Arc(1, 2, (0,), 3),
        ),
        (Boundary(0, 4),),
        (Boundary(2, 4),),
        (0.75, 0.125, 0.125, 0.0625, 1),
    )


def _independent_solution(p):
    # Solve the first-step equations by elimination. The first row
    # has two parallel p contributions and no immediate final mass.
    retained, back, finish = Fraction(7, 8), Fraction(1, 8), Fraction(1, 16)
    determinant = retained**2 - 2 * p * back
    return 2 * p * finish / determinant, 2 * finish * retained**2 / determinant**2


def test_cyclic_sparse_certificate_matches_rational_system_and_reads_the_star_once(
    monkeypatch,
):
    relation, source = _two_state()
    original = queries._close_block
    calls = []

    def counted(block, *args):
        calls.append(block.order)
        return original(block, *args)

    monkeypatch.setattr(queries, "_close_block", counted)
    for value in (Fraction(3, 4), Fraction(5, 8)):
        with torch.no_grad():
            source[0] = float(value)
        exact, derivative = _independent_solution(value)
        proof = _certify(relation, relative_tolerance=_TOL, max_closure_cells=9)
        numeric = Fraction(proof.weight.item())
        assert proof.lower <= exact <= proof.upper
        assert abs(numeric - exact) <= proof.absolute_error <= _TOL
        assert abs(numeric - exact) / exact <= proof.relative_error <= _TOL
        assert proof.closure_cells == 9
        (gradient,) = torch.autograd.grad(proof.weight, source)
        assert gradient[0].item() == pytest.approx(float(derivative))
    assert calls == [3, 3]


def test_certified_score_and_intervals_share_one_call_local_snapshot(monkeypatch):
    relation, source = _two_state()
    exact, _ = _independent_solution(Fraction(3, 4))
    propose = accuracy._propose_suffixes

    def update_owner(closed, rhs):
        with torch.no_grad():
            source[3].fill_(0.125)
        return propose(closed, rhs)

    with monkeypatch.context() as patch:
        patch.setattr(accuracy, "_propose_suffixes", update_owner)
        prior = _certify(relation)
    assert prior.lower <= exact <= prior.upper
    current = _certify(relation)
    assert current.lower <= 2 * exact <= current.upper
    assert current.weight.item() == 2 * prior.weight.item()


def test_multiple_distinct_closures_propagate_in_reverse_component_order():
    relation, _ = _relation(
        3,
        (
            Arc(0, 0, (None,), 0),
            Arc(0, 1, (0,), 2),
            Arc(1, 1, (None,), 1),
            Arc(1, 2, (0,), 2),
        ),
        (Boundary(0, 3),),
        (Boundary(2, 3),),
        (0.5, 0.75, 0.125, 1),
    )
    exact = Fraction(1, 8) ** 2 / (Fraction(1, 2) * Fraction(1, 4))
    proof = _certify(
        relation,
        ((0, 0),),
        absolute_tolerance=0,
        relative_tolerance=0,
        max_closure_cells=2,
    )
    assert proof.lower == proof.upper == proof.weight.item() == exact
    assert proof.closure_cells == 2


def test_resource_checks_precede_closure_and_bound_exact_integer_intermediates(
    monkeypatch,
):
    relation, _ = _two_state()
    proof = _certify(relation)
    again = _certify(
        relation, max_exact_bits=proof.exact_bits, max_exact_work=proof.exact_work
    )
    assert again.lower == proof.lower and again.upper == proof.upper
    for name, cap in (
        ("max_exact_bits", proof.exact_bits),
        ("max_exact_work", proof.exact_work),
    ):
        with pytest.raises(RelationLimitError, match=name):
            _certify(relation, **{name: cap - 1})
    with pytest.raises(RelationLimitError, match="reading weight slots"):
        _certify(relation, max_exact_work=2)

    def unexpected_close(*_args, **_kwargs):
        raise AssertionError(
            "the dense star must not be allocated beyond its cell limit"
        )

    monkeypatch.setattr(queries, "_close_block", unexpected_close)
    with pytest.raises(RelationLimitError, match="max_closure_cells"):
        _certify(relation, max_closure_cells=8)


def test_near_critical_small_residual_cannot_certify_a_bad_suffix(monkeypatch):
    delta = Fraction(1, 1 << 40)
    relation, _ = _relation(
        1,
        (Arc(0, 0, (None,), 0),),
        (Boundary(0, 2),),
        (Boundary(0, 1),),
        (float(1 - delta), float(delta), 1),
    )
    assert abs(delta + (1 - delta) * 2 - 2) == delta < Fraction(1, 10**9)
    proof = _certify(relation, ((),), absolute_tolerance=0, relative_tolerance=0)
    assert proof.lower == proof.upper == Fraction(1)

    def inaccurate_suffix(original, rhs):
        _, witnesses = propose(original, rhs)
        return [2.0], witnesses

    propose = accuracy._propose_suffixes
    monkeypatch.setattr(accuracy, "_propose_suffixes", inaccurate_suffix)
    with pytest.raises(RelationAccuracyError, match="requested absolute path error"):
        _certify(relation, ((),), absolute_tolerance=Fraction(1, 10**9))


def test_contraction_failure_divergence_and_unsupported_arithmetic_are_distinct(
    monkeypatch,
):
    relation, _ = _two_state()
    propose = accuracy._propose_suffixes

    def noncontracting_witness(closed, rhs):
        centers, _ = propose(closed, rhs)
        return centers, [1.0] * len(rhs)

    monkeypatch.setattr(accuracy, "_propose_suffixes", noncontracting_witness)
    with pytest.raises(RelationAccuracyError, match="verify contraction"):
        _certify(relation)

    cyclic = (Arc(0, 0, (None,), 0), Arc(0, 0, (None,), 0))
    starts = (Boundary(0, 1),)
    divergent, _ = _relation(1, cyclic, starts, starts, (0.75, 1))
    with pytest.raises(DivergentClosureError):
        _certify(divergent, ((),))
    logarithmic, _ = _relation(1, (), starts, starts, (-1000, 0), algebra=LOG_PROB)
    with pytest.raises(TypeError, match="requires Markov"):
        _certify(logarithmic, ((),))
    narrow, _ = _relation(1, (), starts, starts, (0.5, 1), dtype=torch.float32)
    with pytest.raises(TypeError, match="requires float64"):
        _certify(narrow, ((),))


def test_impossible_and_zero_live_support_certify_exact_zero():
    relation, _ = _relation(
        2, (Arc(0, 1, (0,), 0),), (Boundary(0, 1),), (Boundary(1, 1),), (0.0, 1)
    )
    for words in (((0,),), ((1,),)):
        proof = _certify(relation, words, absolute_tolerance=0, relative_tolerance=0)
        assert proof.weight.item() == proof.lower == proof.upper == 0
        assert proof.absolute_error == proof.relative_error == 0
