"""Nominal scalar effect equations against formal formulas and strict domains."""

from dataclasses import replace
from fractions import Fraction

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EffectEquation,
    PolynomialLawDecision,
    PolynomialLawLimit,
    PolynomialLawUnsupported,
    QiecModule,
    RationalLawDecision,
    RationalLawLimit,
    RationalLawUnsupported,
    check_scalar_effect_law,
)

_SOURCE = """
effect Scale
    transform : Real -> Real

instance scaling : Scale

handler linear for Scale : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    transform(v : Real) resumes 1 =>
        resume(s * v)

handler polynomial_square for Scale : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    transform(v : Real) resumes 1 =>
        resume(s * v * v)

handler rational for Scale : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    transform(v : Real) resumes 1 =>
        resume(v / s)

handler rational_square for Scale : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    transform(v : Real) resumes 1 =>
        resume(v * v / s)

define combined(a : Real, b : Real) : Real !{scaling} =
    perform scaling.transform(a + b)

define separate(a : Real, b : Real) : Real !{scaling} =
    let first <- perform scaling.transform(a)
    let second <- perform scaling.transform(b)
    return first + second

define guarded(a : Real, b : Real) : Real !{scaling} =
    let unused = 1.0 / (a - b)
    perform scaling.transform(a + b)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(
        parse(_SOURCE), module_name="scalar-effect-law", file_path="scale.qvr"
    )
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _equation(module: QiecModule) -> EffectEquation:
    instance = next(item.entry for item in module.instances if item.name == "scaling")
    interface = next(
        item for item in module.effects if item.ref.id == instance.effect.id
    )
    roots = {item.name: item.id for item in module.computations}
    return EffectEquation(
        interface.ref.id,
        "additivity",
        instance.instance,
        (interface.operations[0].id,),
        roots["combined"],
        roots["separate"],
    )


def test_polynomial_handler_additivity_uses_symbolic_initial_state(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    linear, square, _, _ = module.handlers
    good = check_scalar_effect_law(module, equation, linear.id, method="polynomial")
    assert good.certified
    assert isinstance(good.result, PolynomialLawDecision)
    assert len(good.parameter_types) == 3
    expected = {(0, 1, 1): Fraction(1), (1, 0, 1): Fraction(1)}
    assert dict(good.result.left) == dict(good.result.right) == expected

    bad = check_scalar_effect_law(module, equation, square.id, method="polynomial")
    assert not bad.certified
    witness = bad.result.counterexample
    assert witness is not None
    a, b, scale = map(Fraction, witness.arguments)
    assert a * b * scale != 0
    assert (witness.left, witness.right) == (
        scale * (a + b) ** 2,
        scale * (a * a + b * b),
    )


def test_rational_handler_additivity_preserves_domain_and_exact_witness(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    _, _, linear, square = module.handlers
    good = check_scalar_effect_law(module, equation, linear.id, method="rational")
    assert good.certified
    assert isinstance(good.result, RationalLawDecision)
    assert good.result.domain_polynomial == (((0, 0, 1), Fraction(1)),)

    bad = check_scalar_effect_law(module, equation, square.id, method="rational")
    assert not bad.certified
    assert isinstance(bad.result, RationalLawDecision)
    assert bad.result.domain_polynomial == good.result.domain_polynomial
    witness = bad.result.counterexample
    assert witness is not None
    a, b, scale = map(Fraction, witness.arguments)
    assert a * b * scale != 0
    assert (witness.left, witness.right) == (
        (a + b) ** 2 / scale,
        (a * a + b * b) / scale,
    )


def test_strict_rational_guards_and_polynomial_limit_have_distinct_contracts(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    rational = module.handlers[2]
    guarded = next(item.id for item in module.computations if item.name == "guarded")
    narrowed = equation.with_(left=guarded)
    with pytest.raises(RationalLawUnsupported, match="equal rational law domains"):
        check_scalar_effect_law(module, narrowed, rational.id, method="rational")
    with pytest.raises(PolynomialLawUnsupported):
        check_scalar_effect_law(module, equation, rational.id, method="polynomial")
    with pytest.raises(PolynomialLawLimit, match="work limit"):
        check_scalar_effect_law(
            module, equation, module.handlers[0].id, method="polynomial", max_work=1
        )
    with pytest.raises(RationalLawLimit, match="steps"):
        check_scalar_effect_law(
            module, equation, rational.id, method="rational", fuel=1
        )


def test_prior_binding_keeps_method_handler_and_result_separate(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    linear, _, rational, square = module.handlers
    polynomial = check_scalar_effect_law(
        module, equation, linear.id, method="polynomial"
    )
    with pytest.raises(RationalLawUnsupported, match="evidence does not match"):
        check_scalar_effect_law(
            module, equation, linear.id, method="rational", prior=polynomial
        )
    same_rational = check_scalar_effect_law(
        module, equation, linear.id, method="rational"
    )
    assert same_rational.fingerprint != polynomial.fingerprint
    mixed_receipt = replace(polynomial, fingerprint=same_rational.fingerprint)
    with pytest.raises(RationalLawUnsupported, match="evidence does not match"):
        check_scalar_effect_law(
            module, equation, linear.id, method="rational", prior=mixed_receipt
        )
    with pytest.raises(PolynomialLawUnsupported, match="evidence does not match"):
        check_scalar_effect_law(
            module, equation, rational.id, method="polynomial", prior=polynomial
        )

    bad = check_scalar_effect_law(module, equation, square.id, method="rational")
    assert isinstance(bad.result, RationalLawDecision)
    forged = replace(bad, result=replace(bad.result, counterexample=None))
    assert (
        check_scalar_effect_law(
            module, equation, square.id, method="rational", prior=forged
        )
        == bad
    )
