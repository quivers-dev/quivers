"""Rational derivative certificates against strict domain and value oracles."""

from fractions import Fraction

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    PolynomialLawCounterexample,
    QiecModule,
    RationalLawLimit,
    RationalLawUnsupported,
    check_rational_derivative_rule,
    exact_jvp,
)

_SOURCE = """
define ratio(x : Real, y : Real) : Real !{} =
    return x / y + pow(x, -2.0)

define push(a : Real, b : Real, da : Real, db : Real) : Real !{} =
    return (da * b - a * db) / pow(b, 2.0) - 2.0 * da / pow(a, 3.0)

define pull_first(a : Real, b : Real, seed : Real) : Real !{} =
    return seed / b - 2.0 * seed / pow(a, 3.0)

define pull_second(a : Real, b : Real, seed : Real) : Real !{} =
    return (0.0 - seed * a) / pow(b, 2.0)

define false_push(a : Real, b : Real, da : Real, db : Real) : Real !{} =
    return (da * b - a * db) / pow(b, 2.0) - 2.0 * da / pow(a, 3.0) + da * db

define false_pull(a : Real, b : Real, seed : Real) : Real !{} =
    return seed / b - 2.0 * seed / pow(a, 3.0) + seed * (seed - 1.0) * (seed + 1.0)

define hides_undefined(a : Real, b : Real, da : Real, db : Real) : Real !{} =
    let unused = da / da
    return (da * b - a * db) / pow(b, 2.0) - 2.0 * da / pow(a, 3.0)

define partial_seed(a : Real, b : Real, seed : Real) : Real !{} =
    let unused = seed / seed
    return seed / b - 2.0 * seed / pow(a, 3.0)

define sum_ratio(x : Real, y : Real) : Real !{} =
    return x / (x + y)

define sum_push(x : Real, y : Real, dx : Real, dy : Real) : Real !{} =
    return (dx * y - x * dy) / pow(x + y, 2.0)

define partial_sum_push(x : Real, y : Real, dx : Real, dy : Real) : Real !{} =
    let unused = y / y
    return (dx * y - x * dy) / pow(x + y, 2.0)

define strict_source(a : Real, x : Real) : Real !{} =
    let unused = a / a
    return x * x

define square_push(a : Real, x : Real, da : Real, dx : Real) : Real !{} =
    return 2.0 * x * dx

define bad_square_push(a : Real, x : Real, da : Real, dx : Real) : Real !{} =
    return 2.0 * x * dx + 1.0

define unrelated_guard_push(a : Real, x : Real, da : Real, dx : Real) : Real !{} =
    let unused = 1.0 / (1.0 + x * x)
    return 2.0 * x * dx

define constant() : Real !{} =
    return 7.0

define zero() : Real !{} =
    return 0.0

define spin(x : Real) : Real !{} =
    spin(x)

effect Choice
    choose : Unit -> Bool

instance choice : Choice

handler blend for Choice : Real -> Real with (p : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    choose resumes omega =>
        let first <- resume(false)
        let second <- resume(true)
        return p * first + (1.0 - p) * second

define weighted(x : Real, y : Real, p : Real) : Real !{} =
    handle choice with blend(p) in
        let swapped <- perform choice.choose()
        if swapped then
            return y / x
        else
            return x / y

define weighted_push(x : Real, y : Real, p : Real, dx : Real, dy : Real, dp : Real) : Real !{} =
    return dp * (x / y - y / x) + p * (dx * y - x * dy) / pow(y, 2.0) + (1.0 - p) * (dy * x - y * dx) / pow(x, 2.0)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(
        parse(_SOURCE), module_name="derivative-rules", file_path="rules.qvr"
    )
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _push(x: Fraction, y: Fraction, dx: Fraction, dy: Fraction) -> Fraction:
    """Use explicit partial derivatives of the example's rational function."""
    return (1 / y - 2 / x**3) * dx - x * dy / y**2


def test_checks_both_rules_and_their_transpose_on_the_primal_domain(
    module: QiecModule,
) -> None:
    result = check_rational_derivative_rule(
        module, "ratio", jvp="push", vjp=("pull_first", "pull_second")
    )
    assert result.certified
    assert result.parameters == ("x", "y")
    assert result.domain_polynomial == (((1, 1), Fraction(1)),)
    assert result.jvp is not None
    assert result.vjp is not None
    assert result.jvp.computation == "push"
    assert tuple(rule.computation for rule in result.vjp) == (
        "pull_first",
        "pull_second",
    )
    assert all(not rule.difference_numerator for rule in (result.jvp, *result.vjp))

    x, y, dx, dy, seed = map(Fraction, (2, -3, 4, -5, -2))
    forward = exact_jvp(module, "push", (x, y, dx, dy), (0, 0, 0, 0)).value
    assert forward == _push(x, y, dx, dy)
    adjoints = tuple(
        exact_jvp(module, part.computation, (x, y, seed), (0, 0, 0)).value
        for part in result.vjp
    )
    assert seed * forward == dx * adjoints[0] + dy * adjoints[1]


def test_a_symbolic_seed_exposes_nonlinearity_in_a_rule(module: QiecModule) -> None:
    result = check_rational_derivative_rule(module, "ratio", jvp="false_push")
    assert not result.certified
    assert result.vjp is None
    assert result.jvp is not None
    witness = result.jvp.counterexample
    assert witness is not None
    x, y, dx, dy = map(Fraction, witness.arguments)
    assert x * y * dx * dy != 0
    formal = _push(x, y, dx, dy)
    assert witness == PolynomialLawCounterexample(
        witness.arguments, formal + dx * dy, formal
    )
    assert all(len(exponents) == 4 for exponents, _ in result.jvp.difference_numerator)


def test_a_cotangent_outside_the_unit_grid_refutes_a_reverse_rule(
    module: QiecModule,
) -> None:
    result = check_rational_derivative_rule(
        module, "ratio", vjp=("false_pull", "pull_second")
    )
    assert not result.certified
    assert result.jvp is None
    assert result.vjp is not None
    assert result.vjp[1].counterexample is None
    witness = result.vjp[0].counterexample
    assert witness is not None
    x, y, seed = map(Fraction, witness.arguments)
    assert x * y != 0
    assert seed not in (-1, 0, 1)
    formal = seed * (1 / y - 2 / x**3)
    assert witness == PolynomialLawCounterexample(
        witness.arguments, formal + seed * (seed - 1) * (seed + 1), formal
    )
    assert all(
        len(exponents) == 3 for exponents, _ in result.vjp[0].difference_numerator
    )


def test_transposition_respects_parameter_order(module: QiecModule) -> None:
    result = check_rational_derivative_rule(
        module, "ratio", vjp=("pull_second", "pull_first")
    )
    assert not result.certified
    assert result.vjp is not None
    assert all(part.counterexample is not None for part in result.vjp)


def test_guard_coverage_can_admit_a_higher_power_of_a_nonmonomial(
    module: QiecModule,
) -> None:
    result = check_rational_derivative_rule(module, "sum_ratio", jvp="sum_push")
    assert result.certified
    assert dict(result.domain_polynomial) == {(1, 0): Fraction(1), (0, 1): Fraction(1)}
    with pytest.raises(RationalLawUnsupported, match="partial_sum_push.*primal domain"):
        check_rational_derivative_rule(module, "sum_ratio", jvp="partial_sum_push")
    assert exact_jvp(module, "sum_ratio", (1, 0), (0, 1)).tangent == -1


def test_strict_seed_guards_cannot_be_erased_even_when_unused(
    module: QiecModule,
) -> None:
    with pytest.raises(RationalLawUnsupported, match="hides_undefined.*primal domain"):
        check_rational_derivative_rule(module, "ratio", jvp="hides_undefined")
    with pytest.raises(RationalLawUnsupported, match="partial_seed.*primal domain"):
        check_rational_derivative_rule(
            module, "ratio", vjp=("partial_seed", "pull_second")
        )
    assert exact_jvp(module, "ratio", (1, 2), (0, 1)).tangent == Fraction(-1, 4)
    assert exact_jvp(module, "ratio", (1, 2), (0, 0)).tangent == 0


def test_a_total_rule_does_not_extend_the_strict_primal_domain(
    module: QiecModule,
) -> None:
    result = check_rational_derivative_rule(module, "strict_source", jvp="square_push")
    assert result.certified
    assert result.domain_polynomial == (((1, 0), Fraction(1)),)
    with pytest.raises(RationalLawUnsupported, match="could not establish"):
        check_rational_derivative_rule(
            module, "strict_source", jvp="unrelated_guard_push"
        )


def test_counterexample_satisfies_unused_primal_guards(module: QiecModule) -> None:
    result = check_rational_derivative_rule(
        module, "strict_source", jvp="bad_square_push"
    )
    assert not result.certified
    assert result.jvp is not None
    witness = result.jvp.counterexample
    assert witness is not None
    a, x, _, dx = witness.arguments
    assert a != 0
    assert witness == PolynomialLawCounterexample(
        witness.arguments, Fraction(2 * x * dx + 1), Fraction(2 * x * dx)
    )


def test_symbolic_primal_executes_both_authored_resumptions(module: QiecModule) -> None:
    result = check_rational_derivative_rule(module, "weighted", jvp="weighted_push")
    assert result.certified
    assert result.domain_polynomial == (((1, 1, 0), Fraction(1)),)
    point = (Fraction(2), Fraction(3), Fraction(0))
    assert exact_jvp(module, "weighted", point, (0, 0, 1)).tangent == Fraction(-5, 6)


def test_rule_shapes_and_limits_are_checked_before_certifying(
    module: QiecModule,
) -> None:
    assert check_rational_derivative_rule(
        module, "constant", jvp="zero", vjp=()
    ).certified
    with pytest.raises(ValueError, match="at least one"):
        check_rational_derivative_rule(module, "ratio")
    with pytest.raises(RationalLawUnsupported, match="one rule per primal parameter"):
        check_rational_derivative_rule(module, "ratio", vjp=("pull_first",))
    with pytest.raises(RationalLawUnsupported, match="4 positional parameters"):
        check_rational_derivative_rule(module, "ratio", jvp="pull_first")
    with pytest.raises(RationalLawLimit, match="work limit"):
        check_rational_derivative_rule(module, "ratio", jvp="push", max_work=1)
    with pytest.raises(RationalLawLimit, match="steps"):
        check_rational_derivative_rule(module, "spin", jvp="ratio", fuel=10)
