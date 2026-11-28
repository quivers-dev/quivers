"""Rational laws checked against strict-domain and exact-value oracles."""

from fractions import Fraction

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec.module import QiecModule
from quivers.qiec.polynomial_laws import PolynomialLawCounterexample
from quivers.qiec.rational_laws import (
    RationalLawLimit,
    RationalLawUnsupported,
    check_rational_law,
)

_SOURCE = """
effect Choice
    pick : Unit -> Bool

instance coin : Choice

handler average for Choice : Real -> Real with (p : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    pick resumes omega =>
        let lower <- resume(false)
        let upper <- resume(true)
        return (1.0 - p) * lower + p * upper

define weighted(a : Real, x : Real, y : Real) : Real !{} =
    handle coin with average(1.0 / a) in
        let choice <- perform coin.pick()
        if choice then
            return y
        else
            return x

define explicit_weight(a : Real, x : Real, y : Real) : Real !{} =
    return ((a - 1.0) * x + y) / a

define separate(a : Real, b : Real) : Real !{} =
    return (a / a) * (b / b)

define combined(a : Real, b : Real) : Real !{} =
    return (a * b) / (a * b)

define reciprocal_quotient(a : Real, b : Real) : Real !{} =
    return 1.0 / (a / b)

define strict_reciprocal_quotient(a : Real, b : Real) : Real !{} =
    return b / a + 0.0 / b

define identity(a : Real) : Real !{} =
    return a / a

define double_reciprocal(a : Real) : Real !{} =
    return 2.0 / a

define zero_with_guard(a : Real) : Real !{} =
    return 0.0 / a

define one(a : Real) : Real !{} =
    return 1.0

define unused_partial(a : Real) : Real !{} =
    handle coin with average(0.5) in
        let unused = a / a
        let choice <- perform coin.pick()
        return 1.0

define negative_power(a : Real) : Real !{} =
    return pow(a, -2.0)

define reciprocal_twice(a : Real) : Real !{} =
    return (1.0 / a) * (1.0 / a)

define undefined(a : Real) : Real !{} =
    return 1.0 / (a - a)

define signs(a : Real) : Real !{} =
    if a > 0.0 then
        return a
    else
        return 0.0 - a

define forever(a : Real) : Real !{} =
    forever(a)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(
        parse(_SOURCE), module_name="rational-laws", file_path="rat.qvr"
    )
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def test_symbolic_divisor_survives_an_authored_multishot_handler(
    module: QiecModule,
) -> None:
    result = check_rational_law(module, "weighted", "explicit_weight")
    assert result.domain_polynomial == (((1, 0, 0), Fraction(1)),)
    assert not result.difference_numerator
    assert result.counterexample is None


def test_grouping_and_strict_nested_quotients_preserve_domains(
    module: QiecModule,
) -> None:
    for left, right in (
        ("combined", "separate"),
        ("reciprocal_quotient", "strict_reciprocal_quotient"),
    ):
        result = check_rational_law(module, left, right)
        assert result.domain_polynomial == (((1, 1), Fraction(1)),)
        assert not result.difference_numerator
        assert result.counterexample is None


def test_a_rational_counterexample_satisfies_the_common_domain(
    module: QiecModule,
) -> None:
    result = check_rational_law(module, "identity", "double_reciprocal")
    assert result.domain_polynomial == (((1,), Fraction(1)),)
    for a in range(4):
        if a:
            left, right = Fraction(a, a), Fraction(2, a)
            if left != right:
                assert result.counterexample == PolynomialLawCounterexample(
                    (a,), left, right
                )
                break
    else:
        pytest.fail("the independent oracle found no witness in the common domain")


@pytest.mark.parametrize("partial", ("identity", "unused_partial"))
def test_equal_cross_products_cannot_certify_distinct_domains(
    module: QiecModule, partial: str
) -> None:
    with pytest.raises(RationalLawUnsupported, match="equal rational law domains"):
        check_rational_law(module, partial, "one")


def test_an_unread_division_keeps_its_domain(module: QiecModule) -> None:
    result = check_rational_law(module, "unused_partial", "identity")
    assert result.domain_polynomial == (((1,), Fraction(1)),)
    assert result.counterexample is None


def test_a_witness_keeps_guards_when_both_results_are_constant(
    module: QiecModule,
) -> None:
    result = check_rational_law(module, "zero_with_guard", "unused_partial")
    assert result.domain_polynomial == (((1,), Fraction(1)),)
    assert result.counterexample == PolynomialLawCounterexample(
        (1,), Fraction(0), Fraction(1)
    )


def test_negative_powers_keep_the_nonzero_guard(module: QiecModule) -> None:
    result = check_rational_law(module, "negative_power", "reciprocal_twice")
    assert result.domain_polynomial == (((1,), Fraction(1)),)
    assert result.counterexample is None
    with pytest.raises(RationalLawUnsupported, match="equal rational law domains"):
        check_rational_law(module, "negative_power", "one")


@pytest.mark.parametrize(
    ("name", "reason"),
    (("undefined", "identically zero"), ("signs", "must be constant")),
)
def test_unsupported_rational_domains_cannot_be_certified(
    module: QiecModule, name: str, reason: str
) -> None:
    with pytest.raises(RationalLawUnsupported, match=reason):
        check_rational_law(module, name, name)


def test_rational_limits_cannot_certify_an_equation(module: QiecModule) -> None:
    with pytest.raises(RationalLawLimit, match="work limit"):
        check_rational_law(module, "weighted", "explicit_weight", max_work=2)
    with pytest.raises(RationalLawLimit, match="steps"):
        check_rational_law(module, "forever", "forever", fuel=12)
