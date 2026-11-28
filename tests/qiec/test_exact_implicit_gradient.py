"""Exact implicit gradients against explicit root branches and independent AD."""

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExactImplicitGradient,
    ImplicitRootError,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    QiecModule,
    discharge_module,
    exact_implicit_gradient,
)

_SOURCE = """
define quadratic(z : Real, x : Real, a : Real) : Real !{} =
    return z * z - x * a

define scaled(z : Real, x : Real, a : Real) : Real !{} =
    return (z * z - x * a) / (z + 1.0)

define fixed(z : Real, rate : Real, cost : Real) : Real !{} =
    return z - (rate * z + cost)

define reverse_residual(z : Real, rate : Real, cost : Real) : Real !{} =
    return rate * z + cost - z

define singular(z : Real, x : Real) : Real !{} =
    return z * z * z - x

define branch_boundary(z : Real, x : Real) : Real !{} =
    if z > 0.0 then
        return 2.0 * z - x
    else
        return z - x

define unused_guard(z : Real, x : Real) : Real !{} =
    let unused = z / z
    return z - x

define one_parameter(z : Real) : Real !{} =
    return z - 3.0

define fixed_terms(z : Real, weight : Real, gain : Real, x : Real) : Real !{} =
    return weight * z - gain * x

define forever(z : Real) : Real !{} =
    forever(z)

effect Orbit
    branch : Unit -> Bool

instance orbit : Orbit

handler blend for Orbit : Real -> Real with (p : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    branch resumes omega =>
        let lower <- resume(false)
        let upper <- resume(true)
        return p * lower + (1.0 - p) * upper

define authored(z : Real, x : Real, p : Real) : Real !{} =
    handle orbit with blend(p) in
        let flag <- perform orbit.branch()
        if flag then
            return (z * z - x) / 2.0
        else
            return z * z - x
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(
        parse(_SOURCE), module_name="implicit-roots", file_path="root.qvr"
    )
    assert compiler.qiec_module is not None
    return compiler.qiec_module


@pytest.mark.parametrize("root", (3, -3))
def test_quadratic_branches_agree_with_explicit_root_and_autograd(
    module: QiecModule, root: int
) -> None:
    z, x, a = Fraction(root), Fraction(9, 2), Fraction(2)
    expected = (a / (2 * z), x / (2 * z))
    result = exact_implicit_gradient(module, "quadratic", z, (x, a))
    assert result == ExactImplicitGradient(z, expected, 2 * z)
    assert result.semantics.kind == "implicit"
    assert {condition.name for condition in result.semantics.implicit_conditions} == {
        "regular-residual",
        "zero-residual",
        "nonzero-state-derivative",
    }

    scaled = exact_implicit_gradient(module, "scaled", z, (x, a))
    assert scaled == ExactImplicitGradient(z, expected, 2 * z / (z + 1))

    tx = torch.tensor(float(x), dtype=torch.float64, requires_grad=True)
    ta = torch.tensor(float(a), dtype=torch.float64, requires_grad=True)
    solved = (-1 if z < 0 else 1) * torch.sqrt(tx * ta)
    reference = torch.autograd.grad(solved, (tx, ta))
    assert tuple(float(item) for item in reference) == pytest.approx(
        tuple(map(float, result.gradient)), rel=1e-13
    )

    with localcontext() as context:
        context.prec = 110
        point, height = Decimal("4.5"), Decimal(2)
        step = Decimal(2) ** -115
        upper, lower = (
            (height * (point + step)).sqrt(),
            (height * (point - step)).sqrt(),
        )
        observed = (-1 if z < 0 else 1) * (upper - lower) / (2 * step)
        formal = Decimal(expected[0].numerator) / Decimal(expected[0].denominator)
        assert abs(observed - formal) < Decimal("1e-68")


@pytest.mark.parametrize("sign", (-1, 1))
def test_near_critical_fixed_point_uses_the_actual_nonzero_derivative(
    module: QiecModule, sign: int
) -> None:
    delta = sign * Fraction(1, 2**64)
    rate, cost = 1 - delta, 2 * delta
    expected = (2 / delta, 1 / delta)
    point = (rate, cost)
    result = exact_implicit_gradient(module, "fixed", 2, point)
    assert result == ExactImplicitGradient(Fraction(2), expected, delta)
    assert exact_implicit_gradient(module, "reverse_residual", 2, point) == (
        ExactImplicitGradient(Fraction(2), expected, -delta)
    )

    # The explicit root is cost / (1 - rate).
    step = abs(delta) / 2**180
    difference = (cost / (1 - rate - step) - cost / (1 - rate + step)) / (2 * step)
    assert abs(difference / result.gradient[0] - 1) < Fraction(2, 2**360)


def test_authored_multishot_residual_agrees_before_and_after_discharge(
    module: QiecModule,
) -> None:
    discharged = discharge_module(module)
    assert discharged.obstacle("authored") is None
    for owner in (module, discharged.module):
        assert exact_implicit_gradient(owner, "authored", 3, (9, 0)) == (
            ExactImplicitGradient(
                Fraction(3), (Fraction(1, 6), Fraction(0)), Fraction(3)
            )
        )


def test_unverified_or_singular_roots_are_refused(module: QiecModule) -> None:
    with pytest.raises(ImplicitRootError, match="does not make the residual zero"):
        exact_implicit_gradient(
            module, "quadratic", 3, (Fraction(9, 2) + Fraction(1, 2**128), 2)
        )
    with pytest.raises(ImplicitRootError, match="state derivative is zero"):
        exact_implicit_gradient(module, "singular", 0, (0,))
    with pytest.raises(ImplicitRootError, match="state derivative is zero"):
        exact_implicit_gradient(module, "fixed", 2, (1, 0))


def test_strict_regular_residual_is_required(module: QiecModule) -> None:
    with pytest.raises(JvpBoundaryError, match="input-dependent equality boundary"):
        exact_implicit_gradient(module, "branch_boundary", 0, (0,))
    with pytest.raises(JvpUnsupported, match="division by zero"):
        exact_implicit_gradient(module, "unused_guard", 0, (0,))
    with pytest.raises(JvpLimit, match="steps"):
        exact_implicit_gradient(module, "forever", 0, (), fuel=10)


def test_empty_gradient_and_implicit_coefficient_bound(module: QiecModule) -> None:
    assert exact_implicit_gradient(module, "one_parameter", 3, ()) == (
        ExactImplicitGradient(Fraction(3), (), Fraction(1))
    )
    with pytest.raises(JvpLimit, match="coefficient limit"):
        exact_implicit_gradient(
            module, "fixed_terms", 0, (Fraction(1, 2**9_000), 2**9_000, 0)
        )
