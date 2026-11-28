"""Exact forward derivatives against independent scalar laws."""

from collections.abc import Callable
from fractions import Fraction

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExactJvp,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    discharge_module,
    exact_jvp,
    run_named,
)
from quivers.qiec.module import QiecModule

_SOURCE = """
effect Hop
    move : Real -> Real

instance hop : Hop
instance outer_hop : Hop
instance inner_hop : Hop

handler abort for Hop : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    move(v : Real) resumes 0 =>
        return v * v

handler conditional for Hop : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    move(v : Real) resumes aff =>
        if v > 0.0 then
            resume(v * v)
        else
            return 3.0 * v

handler carry for Hop : Real -> Real with (state : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return state * x
    move(v : Real) resumes 1 =>
        let answer <- resume(v + state) with (v * state)
        return answer + state * v

handler contrast for Hop : Real -> Real with (p : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    move(v : Real) resumes omega =>
        let lower <- resume(v)
        let upper <- resume(0.0 - v)
        return p * lower + (1.0 - p) * upper

handler shift for Hop : Real -> Real with (offset : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    move(v : Real) resumes 1 =>
        resume(v + offset)

define stopped(x : Real) : Real !{} =
    handle hop with abort in
        let y <- perform hop.move(x)
        return y / 0.0

define affined(x : Real) : Real !{} =
    handle hop with conditional in
        let y <- perform hop.move(x)
        return y * y + 0.25 * x

define threaded(x : Real, initial : Real) : Real !{} =
    handle hop with carry(initial) in
        let y <- perform hop.move(x)
        return y * y

define combined(x : Real, p : Real) : Real !{} =
    handle hop with contrast(p) in
        let y <- perform hop.move(x)
        return y * y * y + x * x

define nested(a : Real, b : Real) : Real !{} =
    handle outer_hop with shift(a) in
        handle inner_hop with shift(b) in
            let y <- perform inner_hop.move(a)
            let z <- perform outer_hop.move(y)
            return z * z

define recurse(x : Real) : Real !{} =
    if x > 1.0 then
        let earlier <- recurse(x - 1.0)
        return x * earlier
    else
        return x * x

define quotient(a : Real, b : Real) : Real !{} =
    return a / b + pow(a, -2.0)

define strict(x : Real) : Real !{} =
    return 0.0 * (x / 0.0)

define fixed_branch(x : Real) : Real !{} =
    if 0.0 == 0.0 then
        return x * x
    else
        return x / 0.0

define variable_exponent(a : Real, b : Real) : Real !{} =
    return pow(a, b)

define unsupported(x : Real) : Real !{} =
    return exp(x)

define forever(x : Real) : Real !{} =
    forever(x)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(parse(_SOURCE), module_name="exact-jvp", file_path="jvp.qvr")
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _fixed_polynomial_derivative(
    function: Callable[..., Fraction],
    point: tuple[Fraction, ...],
    direction: tuple[Fraction, ...],
) -> Fraction:
    """Exact centered difference through degree four, on a stable path."""
    step = Fraction(1, 64)

    def at(multiplier: int) -> Fraction:
        return function(
            *(
                x + multiplier * step * dx
                for x, dx in zip(point, direction, strict=True)
            )
        )

    return (at(-2) - 8 * at(-1) + 8 * at(1) - at(2)) / (12 * step)


@pytest.mark.parametrize(
    ("name", "point", "direction", "formula"),
    (
        ("stopped", (Fraction(3, 2),), (Fraction(2, 3),), lambda x: x * x),
        ("affined", (Fraction(3, 2),), (Fraction(2, 3),), lambda x: x**4 + x / 4),
        ("affined", (Fraction(-3, 2),), (Fraction(2, 3),), lambda x: 3 * x),
        (
            "threaded",
            (Fraction(3, 2), Fraction(-2, 3)),
            (Fraction(1, 2), Fraction(4, 5)),
            lambda x, s: x * s * ((x + s) ** 2 + 1),
        ),
        (
            "combined",
            (Fraction(2, 3), Fraction(1, 3)),
            (Fraction(5, 7), Fraction(-2, 3)),
            lambda x, p: (2 * p - 1) * x**3 + x**2,
        ),
        (
            "nested",
            (Fraction(3, 2), Fraction(-2, 3)),
            (Fraction(1, 2), Fraction(4, 5)),
            lambda a, b: (2 * a + b) ** 2,
        ),
        (
            "recurse",
            (Fraction(5, 2),),
            (Fraction(3, 4),),
            lambda x: x * (x - 1) * (x - 2) ** 2,
        ),
    ),
)
def test_exact_jvp_agrees_with_independent_finite_differences(
    module: QiecModule,
    name: str,
    point: tuple[Fraction, ...],
    direction: tuple[Fraction, ...],
    formula: Callable[..., Fraction],
) -> None:
    expected = ExactJvp(
        formula(*point), _fixed_polynomial_derivative(formula, point, direction)
    )
    assert exact_jvp(module, name, point, direction) == expected


def test_exact_quotient_and_negative_power(module: QiecModule) -> None:
    a, b = Fraction(3, 2), Fraction(2, 3)
    da, db = Fraction(-2, 5), Fraction(5, 6)
    expected = ExactJvp(
        a / b + 1 / a**2,
        da / b - a * db / b**2 - 2 * da / a**3,
    )
    assert exact_jvp(module, "quotient", (a, b), (da, db)) == expected


def test_the_tangent_is_linear_in_the_supplied_direction(module: QiecModule) -> None:
    point = (Fraction(3, 2), Fraction(2, 3))
    first = exact_jvp(module, "combined", point, (2, 0))
    second = exact_jvp(module, "combined", point, (0, 3))
    total = exact_jvp(module, "combined", point, (2, 3))
    assert first.value == second.value == total.value
    assert total.tangent == first.tangent + second.tangent


def test_zero_current_weight_keeps_its_tangent(module: QiecModule) -> None:
    x = Fraction(3, 2)
    expected = ExactJvp(x**2 - x**3, 2 * x**3)
    discharged = discharge_module(module)
    assert discharged.obstacle("combined") is None
    for owner in (module, discharged.module):
        assert exact_jvp(owner, "combined", (x, 0), (0, 1)) == expected


def test_discharge_preserves_exact_jvp_and_agrees_with_reverse_mode(
    module: QiecModule,
) -> None:
    point = (Fraction(3, 2), Fraction(2, 3))
    direction = (Fraction(2, 5), Fraction(-3, 7))
    exact = exact_jvp(module, "combined", point, direction)
    discharged = discharge_module(module)
    assert discharged.obstacle("combined") is None
    assert exact_jvp(discharged.module, "combined", point, direction) == exact

    inputs = tuple(
        torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
        for value in point
    )
    runtime = run_named(discharged.module, "combined", inputs, record_trace=False).value
    assert isinstance(runtime, torch.Tensor)
    cotangents = torch.autograd.grad(runtime, inputs)
    reverse = sum(
        float(gradient) * float(step)
        for gradient, step in zip(cotangents, direction, strict=True)
    )
    assert float(runtime.detach()) == pytest.approx(float(exact.value), rel=1e-13)
    assert reverse == pytest.approx(float(exact.tangent), rel=1e-13)


@pytest.mark.parametrize(("name", "point"), (("affined", 0), ("recurse", 2)))
def test_a_zero_direction_does_not_certify_a_branch_boundary(
    module: QiecModule, name: str, point: int
) -> None:
    with pytest.raises(JvpBoundaryError, match="input-dependent equality boundary"):
        exact_jvp(module, name, (point,), (0,))


def test_a_constant_comparison_does_not_execute_the_inactive_branch(
    module: QiecModule,
) -> None:
    assert exact_jvp(module, "fixed_branch", (3,), (2,)) == ExactJvp(
        Fraction(9), Fraction(12)
    )


@pytest.mark.parametrize(
    ("name", "point", "direction", "reason"),
    (
        ("strict", (2,), (1,), "division by zero"),
        ("quotient", (0, 1), (1, 0), "negative power of zero"),
        ("variable_exponent", (2, 3), (0, 1), "fixed integral exponent"),
        ("unsupported", (1,), (1,), "primitive `exp`"),
    ),
)
def test_no_derivative_is_silently_invented(
    module: QiecModule,
    name: str,
    point: tuple[int, ...],
    direction: tuple[int, ...],
    reason: str,
) -> None:
    with pytest.raises(JvpUnsupported, match=reason):
        exact_jvp(module, name, point, direction)


def test_fuel_exhaustion_cannot_report_a_derivative(module: QiecModule) -> None:
    with pytest.raises(JvpLimit, match="steps"):
        exact_jvp(module, "forever", (1,), (1,), fuel=10)
