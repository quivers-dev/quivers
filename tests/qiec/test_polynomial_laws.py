"""Exact polynomial decisions against independent rational evaluation."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec.discharge import discharge_module
from quivers.qiec.module import QiecModule
from quivers.qiec.polynomial_laws import (
    PolynomialLawCounterexample,
    PolynomialLawLimit,
    PolynomialLawUnsupported,
    check_polynomial_law,
)
from quivers.qiec.terms import Handle, LetValue, Return

_SOURCE = """
effect BitChoice
    pick : Unit -> Bool

instance coin : BitChoice

handler average for BitChoice : Real -> Real with (p : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    pick resumes omega =>
        let lower <- resume(false)
        let upper <- resume(true)
        return (1.0 - p) * lower + p * upper

define forward(p : Real, low : Real, high : Real) : Real !{} =
    handle coin with average(p) in
        let branch <- perform coin.pick()
        if branch then
            return high
        else
            return low

define half_left(a : Real, b : Real) : Real !{} =
    forward(1.0 / 2.0, a, b)

define half_right(a : Real, b : Real) : Real !{} =
    forward(1.0 / 2.0, b, a)

define third_left(a : Real, b : Real) : Real !{} =
    forward(1.0 / 3.0, a, b)

define third_right(a : Real, b : Real) : Real !{} =
    forward(1.0 / 3.0, b, a)

define identity(a : Real) : Real !{} =
    return 1.0

define partial(a : Real) : Real !{} =
    return a / a

define strict_partial(a : Real) : Real !{} =
    return 0.0 * (a / a)

define unused_partial(a : Real) : Real !{} =
    handle coin with average(0.5) in
        let unused = a / a
        let chosen <- perform coin.pick()
        return 1.0

define zero_scalar(a : Real) : Real !{} =
    return 0.0

define zero_divisor(a : Real) : Real !{} =
    return 1.0 / (a - a)

define nonintegral_power(a : Real) : Real !{} =
    return pow(a, 0.5)

define conditioned(a : Real) : Real !{} =
    if a > 0.0 then
        return a
    else
        return 0.0 - a

define binomial_power(a : Real, b : Real) : Real !{} =
    return pow(a + b, 2.0)

define binomial_expansion(a : Real, b : Real) : Real !{} =
    return a * a + 2.0 * a * b + b * b

define many_roots(a : Real, b : Real) : Real !{} =
    return a * (a - 1.0) * (a - 2.0) * b

define zero(a : Real, b : Real) : Real !{} =
    return 0.0

define high_degree(a : Real) : Real !{} =
    return pow(a, 1025.0)

define forever(a : Real) : Real !{} =
    forever(a)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(
        parse(_SOURCE), module_name="polynomial-laws", file_path="poly.qvr"
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _choice(upper_weight: Fraction, lower: int, upper: int) -> Fraction:
    return (1 - upper_weight) * lower + upper_weight * upper


def test_a_half_weighted_choice_commutes(module: QiecModule) -> None:
    result = check_polynomial_law(module, "half_left", "half_right")
    expected = (((0, 1), Fraction(1, 2)), ((1, 0), Fraction(1, 2)))
    assert result.left == expected == result.right
    assert result.counterexample is None


def test_a_third_weighted_choice_has_an_exact_witness(module: QiecModule) -> None:
    result = check_polynomial_law(module, "third_left", "third_right")
    weight = Fraction(1, 3)
    for first, second in product((0, 1), repeat=2):
        left, right = _choice(weight, first, second), _choice(weight, second, first)
        if left != right:
            assert result.counterexample == PolynomialLawCounterexample(
                (first, second), left, right
            )
            break
    else:
        pytest.fail("the independent oracle found no distinguishing choice")
    assert result.left == (((0, 1), weight), ((1, 0), 1 - weight))
    assert result.right == (((0, 1), 1 - weight), ((1, 0), weight))


def test_multiplication_keeps_the_cross_term(module: QiecModule) -> None:
    result = check_polynomial_law(module, "binomial_power", "binomial_expansion")
    assert result.counterexample is None
    expected = (((0, 2), Fraction(1)), ((1, 1), Fraction(2)), ((2, 0), Fraction(1)))
    assert result.left == expected == result.right


def test_the_witness_search_keeps_a_nonzero_polynomial(module: QiecModule) -> None:
    result = check_polynomial_law(module, "many_roots", "zero")
    for first, second in product(range(4), (0, 1)):
        value = first * (first - 1) * (first - 2) * second
        if value:
            assert result.counterexample == PolynomialLawCounterexample(
                (first, second), Fraction(value), Fraction(0)
            )
            break
    else:
        pytest.fail("the independent grid found no separating assignment")


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    (
        ("partial", "identity", "nonzero constant divisor"),
        ("strict_partial", "zero_scalar", "nonzero constant divisor"),
        ("zero_divisor", "identity", "nonzero constant divisor"),
        ("nonintegral_power", "identity", "integral constant exponent"),
        ("conditioned", "conditioned", "literal branch conditions"),
        ("forever", "forever", "recursive polynomial"),
    ),
)
def test_polynomial_laws_preserve_domain_requirements(
    module: QiecModule, left: str, right: str, reason: str
) -> None:
    with pytest.raises(PolynomialLawUnsupported, match=reason):
        check_polynomial_law(module, left, right)


def test_discharge_cannot_hide_an_unread_partial_value(module: QiecModule) -> None:
    original = next(
        item for item in module.computations if item.name == "unused_partial"
    )
    discharged = next(
        item
        for item in discharge_module(module).module.computations
        if item.name == "unused_partial"
    )
    assert isinstance(original.body, Handle)
    assert isinstance(discharged.body, Return)
    assert not isinstance(discharged.body.value, LetValue)
    with pytest.raises(PolynomialLawUnsupported, match="nonzero constant divisor"):
        check_polynomial_law(module, "unused_partial", "identity")


def test_polynomial_limits_cannot_certify_a_law(module: QiecModule) -> None:
    with pytest.raises(PolynomialLawLimit, match="work limit"):
        check_polynomial_law(module, "half_left", "half_right", max_work=1)
    with pytest.raises(PolynomialLawLimit, match="exponent limit"):
        check_polynomial_law(module, "high_degree", "high_degree")
