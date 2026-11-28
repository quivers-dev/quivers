"""Apply only exact derivative actions belonging to the checked source."""

from fractions import Fraction

import pytest

from quivers.dsl import loads as load_source
from quivers.qiec import (
    ExactGradient,
    ExactJvp,
    RationalDerivativeDeclaration,
    check_rational_derivative_rule,
    exact_gradient,
    exact_jvp,
)
from quivers.qiec import differentiation
from quivers.qiec.differentiation import JvpLimit, JvpUnsupported
from quivers.qiec.module import QiecModule
from quivers.qiec.serialization import dumps, loads


_SOURCE = """
define reciprocal(x : Real) : Real !{} =
    return 1.0 / x

define reciprocal_forward(x : Real, step : Real) : Real !{} =
    return -step / (x * x)

define reciprocal_reverse(x : Real, seed : Real) : Real !{} =
    return -seed / (x * x)

define unrelated(x : Real) : Real !{} =
    return x
"""


def _module(source: str) -> QiecModule:
    module = load_source(source).qiec
    assert isinstance(module, QiecModule)
    return module


def _declaration(module: QiecModule) -> RationalDerivativeDeclaration:
    ids = {entry.name: entry.id for entry in module.computations}
    return RationalDerivativeDeclaration(
        primal=ids["reciprocal"],
        jvp=ids["reciprocal_forward"],
        vjp=(ids["reciprocal_reverse"],),
    )


def test_exact_modes_execute_the_named_actions_after_round_trip(monkeypatch):
    module = _module(_SOURCE)
    rule = loads(dumps(_declaration(module)))
    assert isinstance(rule, RationalDerivativeDeclaration)
    assert check_rational_derivative_rule(module, rule).certified
    with pytest.raises(ValueError, match="already specifies"):
        check_rational_derivative_rule(module, rule, jvp="reciprocal_forward")
    recorded = []
    run = differentiation._scalar_value

    def observe(module, name, arguments, fuel):
        recorded.append((name, arguments))
        return run(module, name, arguments, fuel)

    monkeypatch.setattr(differentiation, "_scalar_value", observe)
    point = (Fraction(2, 3),)
    direct = exact_jvp(module, "reciprocal", point, (2,))
    assert exact_jvp(module, "reciprocal", point, (2,), rule=rule) == direct
    assert direct == ExactJvp(Fraction(3, 2), Fraction(-9, 2))
    assert exact_gradient(module, "reciprocal", point, rule=rule) == ExactGradient(
        Fraction(3, 2), (Fraction(-9, 4),)
    )
    assert recorded == [
        ("reciprocal", point),
        ("reciprocal_forward", (*point, 2)),
        ("reciprocal", point),
        ("reciprocal_reverse", (*point, 1)),
    ]


def test_stale_source_does_not_inherit_a_certificate():
    original = _module(_SOURCE)
    rule = _declaration(original)
    changed = _module(_SOURCE.replace("return -step /", "return step /"))
    assert _declaration(changed) == rule
    decision = check_rational_derivative_rule(changed, rule)
    assert not decision.certified and decision.jvp is not None
    assert decision.jvp.counterexample is not None
    with pytest.raises(JvpUnsupported, match="differs from the exact derivative"):
        exact_jvp(changed, "reciprocal", (2,), (1,), rule=rule)
    with pytest.raises(JvpUnsupported, match="another computation"):
        exact_jvp(original, "unrelated", (2,), (1,), rule=rule)


def test_declared_action_keeps_arity_domain_and_budgets():
    module = _module(_SOURCE)
    rule = _declaration(module)
    with pytest.raises(JvpUnsupported, match="no forward action"):
        exact_jvp(module, "reciprocal", (2,), (1,), rule=rule.with_(jvp=None))
    with pytest.raises(JvpUnsupported, match="no reverse action"):
        exact_gradient(module, "reciprocal", (2,), rule=rule.with_(vjp=None))
    with pytest.raises(JvpLimit):
        exact_jvp(module, "reciprocal", (2,), (1,), rule=rule, max_rule_work=1)
    with pytest.raises(ValueError, match="arity"):
        exact_jvp(module, "reciprocal", (2,), (), rule=rule)
    with pytest.raises(JvpUnsupported, match="division by zero"):
        exact_gradient(module, "reciprocal", (0,), rule=rule)
