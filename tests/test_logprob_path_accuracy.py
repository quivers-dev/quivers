"""Independent rational envelopes and numeric refusal tests for LogProb paths."""

from decimal import Decimal
from fractions import Fraction

import pytest
import torch

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationAccuracyError,
    RelationLimitError,
    RelationTopology,
    certify_logprob_path_weight,
    log_accuracy,
)
from quivers.core.relations._packed import _PackedPaths
from quivers.core.relations.accuracy import _RationalBudget

_TAPE = StringSet(alphabet=FinSet(name="Bits", cardinality=2))
_OPTIONS = dict(
    absolute_log_tolerance=Fraction(1, 10**12),
    max_states=32,
    max_work=64,
    decimal_precision=40,
    max_decimal_exponent=2048,
    max_decimal_work=256,
    max_exact_bits=4096,
    max_exact_work=32,
)


def _relation(states, arcs, initial, final, values, *, dtype=torch.float64):
    topology = RelationTopology(
        (_TAPE,),
        0,
        states,
        tuple(arcs),
        tuple(initial),
        tuple(final),
        tuple(f"w{slot}" for slot in range(len(values))),
    )
    weights = torch.nn.Parameter(torch.tensor(values, dtype=dtype))
    return RationalRelation(topology, weights, algebra=LOG_PROB), weights


def _parallel():
    return _relation(
        3,
        (Arc(0, 1, (0,), 0), Arc(0, 1, (0,), 0), Arc(2, 1, (0,), 4)),
        (Boundary(0, 1), Boundary(2, 3)),
        (Boundary(1, 2),),
        (-1000.0, -3.0, -5.0, -torch.inf, -1e100),
    )


def _certify(relation, words=((0,),), **options):
    return certify_logprob_path_weight(relation, words, **(_OPTIONS | options))


def _log_two_interval():
    # log(2) = 2*atanh(1/3). Bound the omitted positive Taylor series
    # by dropping its increasing denominators after the first omitted term.
    x = Fraction(1, 3)
    m = 96
    lower = 2 * sum((x ** (2 * k + 1) / (2 * k + 1) for k in range(m)), Fraction())
    return lower, lower + 2 * x ** (2 * m + 1) / ((2 * m + 1) * (1 - x * x))


def test_both_endpoints_enclose_independent_series_at_log_minus_one_thousand():
    relation, source = _parallel()
    log_two = _log_two_interval()
    for current in (-1000, -1200):
        with torch.no_grad():
            source[0].fill_(current)
        proof = _certify(relation)
        assert torch.equal(proof.weight, relation.weight(((0,),)))
        expected_low, expected_high = (current - 8 + bound for bound in log_two)
        assert proof.lower <= expected_low <= expected_high <= proof.upper
        measured = Fraction(proof.weight.item())
        assert proof.absolute_log_error == max(
            abs(measured - proof.lower), abs(measured - proof.upper)
        )
        assert (
            max(abs(measured - expected_low), abs(measured - expected_high))
            <= proof.absolute_log_error
        )
        assert proof.absolute_log_error <= _OPTIONS["absolute_log_tolerance"]
        assert proof.upper - proof.lower < Fraction(1, 10**33)
        assert proof.decimal_work == 22 and proof.exact_work == 8
        assert torch.autograd.grad(proof.weight, source)[0].tolist() == pytest.approx(
            [1, 1, 1, 0, 0]
        )


def test_explicit_budgets_and_invalid_packed_result_refuse(monkeypatch):
    relation, _ = _parallel()
    actual = _certify(relation)
    exact_limits = dict(
        max_decimal_work=actual.decimal_work,
        max_exact_work=actual.exact_work,
        max_exact_bits=actual.exact_bits,
    )
    assert _certify(relation, **exact_limits).lower == actual.lower
    for name, limit in exact_limits.items():
        with pytest.raises(RelationLimitError, match=name):
            _certify(relation, **{name: limit - 1})
    with pytest.raises(RelationAccuracyError, match="absolute log path error"):
        _certify(relation, decimal_precision=8)

    evaluate = _PackedPaths.evaluate

    def corrupt(plan, *args, **kwargs):
        return evaluate(plan, *args, **kwargs) + 1

    monkeypatch.setattr(_PackedPaths, "evaluate", corrupt)
    with pytest.raises(RelationAccuracyError, match="absolute log path error"):
        _certify(relation)


def test_certificate_and_float_score_use_one_evaluated_snapshot(monkeypatch):
    relation, source = _parallel()
    exp = log_accuracy._DirectedDecimal.exp

    def change_live_owner(context, log_weight):
        with torch.no_grad():
            source[0].fill_(-1200)
        return exp(context, log_weight)

    with monkeypatch.context() as patch:
        patch.setattr(log_accuracy._DirectedDecimal, "exp", change_live_owner)
        old = _certify(relation)
    updated = _certify(relation)
    lower, upper = _log_two_interval()
    assert old.lower <= -1008 + lower <= -1008 + upper <= old.upper
    assert updated.lower <= -1208 + lower <= -1208 + upper <= updated.upper


def test_tape_observation_unfolds_emitting_loops_but_refuses_epsilon_cycles():
    boundary = (Boundary(0, 1),)
    emitted, _ = _relation(1, (Arc(0, 0, (0,), 0),), boundary, boundary, (-1000.0, 0.0))
    proof = _certify(emitted, ((0, 0),))
    assert proof.lower < -2000 < proof.upper
    epsilon, _ = _relation(
        1, (Arc(0, 0, (None,), 0),), boundary, boundary, (-1000.0, 0.0)
    )
    with pytest.raises(ProductiveEpsilonCycleError):
        _certify(epsilon, ((),))
    with pytest.raises(RelationLimitError, match="max_decimal_exponent"):
        _certify(emitted, ((0,),), max_decimal_exponent=499)
    with pytest.raises(
        RelationLimitError, match="accumulation exceeds max_decimal_exponent"
    ):
        _certify(emitted, ((0, 0),), max_decimal_exponent=500)


def test_decimal_logarithm_and_neighbor_refuse_at_range_boundary():
    relation, _ = _relation(
        2, (Arc(0, 1, (0,), 0),), (Boundary(0, 1),), (Boundary(1, 1),), (2.0**-40, 0.0)
    )
    with pytest.raises(
        RelationLimitError, match="Decimal logarithm exceeds max_decimal_exponent"
    ):
        _certify(relation, max_decimal_exponent=4)
    context = log_accuracy._DirectedDecimal(2, 1, 1)
    with pytest.raises(
        RelationLimitError, match="neighbor exceeds max_decimal_exponent"
    ):
        context._neighbor(Decimal("0.10"), lower=True, name="logarithm")


def test_decimal_accumulation_rounds_one_exact_update_in_both_directions():
    context = log_accuracy._DirectedDecimal(2, 4, 2)
    total = Decimal("0.011"), Decimal("0.011")
    factor = Decimal("0.15"), Decimal("0.15")
    low, high = context.accumulate(total, factor, factor)
    exact = Fraction(11, 1000) + Fraction(15, 100) ** 2
    assert (
        Fraction(low)
        == Fraction(33, 1000)
        < exact
        < Fraction(high)
        == Fraction(34, 1000)
    )


def test_rational_bit_preflight_runs_before_materializing_large_power_of_ten():
    class ConversionMustNotRun(Decimal):
        def as_integer_ratio(self):
            raise AssertionError("oversize rational conversion was attempted")

    for value in ("1e-100000", "1e100000"):
        with pytest.raises(RelationLimitError, match="max_exact_bits"):
            log_accuracy._rational(
                ConversionMustNotRun(value), _RationalBudget(4096, 2)
            )


def test_exact_zero_and_unsupported_precision_have_separate_outcomes():
    relation, source = _parallel()
    with torch.no_grad():
        source[0].fill_(-torch.inf)
    for words in (((0,),), ((1,),)):
        result = _certify(relation, words, max_decimal_work=0, absolute_log_tolerance=0)
        assert result.lower is result.upper is None
        assert result.weight.item() == -torch.inf and result.absolute_log_error == 0
        assert result.decimal_work == 0
    marks = (Boundary(0, 0),)
    narrow, _ = _relation(1, (), marks, marks, (0.0,), dtype=torch.float32)
    with pytest.raises(TypeError, match="requires float64"):
        _certify(narrow, ((),))
