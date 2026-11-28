"""Certify cyclic log scores with independent analytic interval references."""

from fractions import Fraction
from math import exp, expm1, factorial, log

import pytest
import torch

from quivers.core.algebras import DivergentClosureError
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RelationAccuracyError,
    RelationLimitError,
    certify_cyclic_logprob_path_weight,
)
from quivers.core.relations._packed import _PackedPaths
from quivers.core.relations.queries import _PathPlan
from tests.test_logprob_path_accuracy import _certify as _certify_dag
from tests.test_logprob_path_accuracy import _log_two_interval, _relation

_OPTIONS = dict(
    absolute_log_tolerance=Fraction(1, 10**12),
    max_states=32,
    max_work=64,
    max_closure_size=3,
    max_closure_cells=9,
    decimal_precision=42,
    max_decimal_exponent=2048,
    max_decimal_work=256,
    max_exact_bits=32768,
    max_exact_work=256,
)


def _certify(relation, **options):
    return certify_cyclic_logprob_path_weight(relation, ((0,),), **(_OPTIONS | options))


def _parallel_cycle():
    return _relation(
        3,
        (
            Arc(0, 1, (None,), 0),
            Arc(0, 1, (None,), 0),
            Arc(1, 0, (None,), 1),
            Arc(0, 2, (0,), 2),
        ),
        (Boundary(0, 3),),
        (Boundary(2, 3),),
        (-0.375, -1.625, -1000.0, 0.0),
    )


def _self_cycle(log_loop=-(2.0**-40)):
    return _relation(
        2,
        (Arc(0, 0, (None,), 0), Arc(0, 1, (0,), 1)),
        (Boundary(0, 2),),
        (Boundary(1, 2),),
        (log_loop, -1000.0, 0.0),
    )


def _negative_log_one_minus(z, terms):
    # For 0 <= z < 1, the omitted positive terms are bounded by
    # replacing their increasing denominators by the first omitted one.
    power = z
    total = Fraction(0)
    for index in range(1, terms + 1):
        total += power / index
        power *= z
    return total, total + power / ((terms + 1) * (1 - z))


def _parallel_reference():
    # Visiting the second state and returning produces one of two
    # alternatives of exactly log weight -2. This counts returns
    # directly and never uses the observation graph or matrix solver.
    exp_upper = sum((Fraction((-2) ** k, factorial(k)) for k in range(65)))
    exp_lower = exp_upper + Fraction((-2) ** 65, factorial(65))
    lower, _ = _negative_log_one_minus(2 * exp_lower, 92)
    _, upper = _negative_log_one_minus(2 * exp_upper, 92)
    return -1000 + lower, -1000 + upper


def _near_critical_reference():
    # Let delta = 2^-40 and u = 1 - (1 - exp(-delta))/delta.
    # The log mass is -1000 + 40 log 2 - log(1-u).
    # Enclose u by consecutive alternating Taylor sums.
    delta = Fraction(1, 2**40)
    u_lower = sum(((-1) ** (j + 1) * delta**j / factorial(j + 1) for j in range(1, 9)))
    u_upper = u_lower + delta**9 / factorial(10)
    lower, _ = _negative_log_one_minus(u_lower, 8)
    _, upper = _negative_log_one_minus(u_upper, 8)
    log_two_lower, log_two_upper = _log_two_interval()
    return -1000 + 40 * log_two_lower + lower, -1000 + 40 * log_two_upper + upper


def _assert_bounds(proof, endpoints, *, max_width=Fraction(1, 10**35)):
    low, high = endpoints
    assert proof.lower <= low <= high <= proof.upper
    assert proof.upper - proof.lower < max_width
    returned = Fraction(proof.weight.item())
    assert (
        max(abs(returned - low), abs(returned - high))
        <= proof.absolute_log_error
        == max(abs(returned - proof.lower), abs(returned - proof.upper))
        <= _OPTIONS["absolute_log_tolerance"]
    )


def test_parallel_cycle_enclosure_at_minus_one_thousand_and_normal_gradient():
    relation, source = _parallel_cycle()
    proof = _certify(relation)
    _assert_bounds(proof, _parallel_reference())
    assert torch.equal(
        proof.weight, relation.query(((0,),), max_closure_size=3).weight()
    )
    z = 2 * exp(-2)
    repeated = z / (1 - z)
    assert torch.autograd.grad(proof.weight, source)[0].tolist() == pytest.approx(
        [repeated, repeated, 1, 2]
    )
    assert 2 * exp(-0.375) > 1
    # The native closure has one auxiliary state for parallel arcs.
    with pytest.raises(RelationLimitError, match="max_closure_size"):
        _certify(relation, max_closure_size=2)
    with pytest.raises(RelationLimitError, match="max_closure_cells"):
        _certify(relation, max_closure_cells=8)
    for name, used in (
        ("max_decimal_work", proof.decimal_work),
        ("max_exact_work", proof.exact_work),
        ("max_exact_bits", proof.exact_bits),
    ):
        assert _certify(relation, **{name: used}).lower == proof.lower
        with pytest.raises(RelationLimitError, match=name):
            _certify(relation, **{name: used - 1})


def test_near_critical_score_and_small_absolute_residual_mutant(monkeypatch):
    relation, _ = _self_cycle()
    proof = _certify(relation)
    expected = _near_critical_reference()
    _assert_bounds(proof, expected, max_width=Fraction(1, 10**28))
    narrower = _certify(relation, decimal_precision=52)
    _assert_bounds(narrower, expected)
    assert torch.equal(narrower.weight, proof.weight)

    # Doubling the candidate mass leaves an absolute residual exp(-1000),
    # which is zero in float64. Its log error is still log(2).
    evaluate = _PackedPaths.evaluate

    def double_mass(plan, *args, **kwargs):
        return evaluate(plan, *args, **kwargs) + log(2.0)

    monkeypatch.setattr(_PackedPaths, "evaluate", double_mass)
    with pytest.raises(RelationAccuracyError, match="absolute log path error"):
        _certify(relation)


def test_nonfinite_upper_star_refuses_before_ordinary_packed_evaluation(monkeypatch):
    relation, _ = _self_cycle()
    assert torch.isfinite(relation.query(((0,),), max_closure_size=1).weight())

    def ordinary_score_must_not_run(*args, **kwargs):
        raise AssertionError(
            "the ordinary score ran without a finite upper-star witness"
        )

    monkeypatch.setattr(_PathPlan, "evaluate", ordinary_score_must_not_run)
    # The true star is finite, but the directed upper return endpoint
    # exceeds one at this precision. No claim of actual divergence follows.
    with pytest.raises(RelationAccuracyError, match="finite upper star"):
        _certify(relation, decimal_precision=8)
    actual_divergence, _ = _self_cycle(log_loop=0)
    with pytest.raises(DivergentClosureError, match="lower return mass"):
        _certify(actual_divergence)


def test_two_components_transfer_tiny_mass_and_share_the_original_parameter():
    delta = 2.0**-40
    relation, source = _relation(
        2,
        (
            Arc(0, 0, (None,), 0),
            Arc(0, 1, (0,), 1),
            Arc(1, 1, (None,), 0),
        ),
        (Boundary(0, 2),),
        (Boundary(1, 2),),
        (-delta, -1000.0, 0.0),
    )
    proof = _certify(relation, decimal_precision=52, max_closure_cells=2)
    expected = tuple(2 * bound + 1000 for bound in _near_critical_reference())
    _assert_bounds(proof, expected)
    assert torch.autograd.grad(proof.weight, source)[0].tolist() == pytest.approx(
        [2 * exp(-delta) / -expm1(-delta), 1, 2]
    )
    with pytest.raises(RelationLimitError, match="max_closure_cells"):
        _certify(relation, max_closure_cells=1)
    with torch.no_grad():
        source[1].fill_(-torch.inf)
    absent = _certify(relation)
    assert absent.lower is absent.upper is None
    assert absent.weight.item() == -torch.inf and absent.absolute_log_error == 0


def test_acyclic_api_still_refuses_the_same_productive_cycles():
    relation, _ = _self_cycle()
    with pytest.raises(ProductiveEpsilonCycleError, match="productive epsilon cycle"):
        _certify_dag(relation)
