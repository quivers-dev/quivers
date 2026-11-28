"""Nominal effect-law obligation records and their checked links."""

import didactic.api as dx
import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Condition,
    EffectEquation,
    EffectLawBundle,
    EffectLawObligation,
    HandlerId,
    OperationId,
)

_SOURCE = """
effect Reader
    get : Unit -> Bool

instance reader : Reader

handler fixed for Reader : Bool -> Bool with (value : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(value)

handler flipped for Reader : Bool -> Bool with (value : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        if value then
            resume(false)
        else
            resume(true)

define first() : Bool !{reader} =
    perform reader.get()

define second() : Bool !{reader} =
    let value <- perform reader.get()
    return value
"""


@pytest.fixture(scope="module")
def bundle() -> EffectLawBundle:
    module = Compiler(parse(_SOURCE), module_name="effect-law-records").qiec_module
    assert module is not None
    row = next(item.entry for item in module.instances if item.name == "reader")
    interface = next(item for item in module.effects if item.ref.id == row.effect.id)
    left, right = module.computations
    equation = EffectEquation(
        interface.ref.id,
        "read-equality",
        row.instance,
        (interface.operations[0].id,),
        left.id,
        right.id,
    )
    obligation = EffectLawObligation(
        equation=equation,
        handler=module.handlers[0].id,
        method="finite",
        guards=(
            Condition(name="state", statement="The reader has Boolean state."),
            Condition(name="row", statement="The reader is the whole residual row."),
        ),
    )
    return EffectLawBundle(module=module, obligations=(obligation,))


def test_bundle_normalizes_obligation_and_guard_order(bundle: EffectLawBundle) -> None:
    original = bundle.obligations[0]
    assert tuple(guard.name for guard in original.guards) == ("row", "state")
    other = original.with_(handler=bundle.module.handlers[1].id)
    assert bundle.with_(obligations=(original, other)) == bundle.with_(
        obligations=(other, original)
    )
    assert original == original.with_(guards=original.guards[::-1])


def test_bundle_rejects_duplicate_checks_and_competing_equations(
    bundle: EffectLawBundle,
) -> None:
    original = bundle.obligations[0]
    with pytest.raises(dx.ValidationError, match="duplicate effect law obligation"):
        bundle.with_(obligations=(original, original))
    changed = original.with_(
        equation=original.equation.with_(right=original.equation.left),
        handler=bundle.module.handlers[1].id,
    )
    with pytest.raises(
        dx.ValidationError, match="equations reuse a nominal effect law name"
    ):
        bundle.with_(obligations=(original, changed))


def test_bundle_rejects_a_missing_nominal_handler_or_operation(
    bundle: EffectLawBundle,
) -> None:
    original = bundle.obligations[0]
    alien_handler = HandlerId.derive("unrelated", "fixed")
    alien_operation = OperationId.derive("unrelated", "get")
    for mismatched in (
        original.with_(handler=alien_handler),
        original.with_(equation=original.equation.with_(operations=(alien_operation,))),
    ):
        with pytest.raises(dx.ValidationError, match="mismatched module declaration"):
            bundle.with_(obligations=(mismatched,))


def test_obligation_rejects_malformed_and_duplicate_guards(
    bundle: EffectLawBundle,
) -> None:
    original = bundle.obligations[0]
    guard = original.guards[0]
    with pytest.raises(dx.ValidationError, match="duplicate effect law guard name"):
        original.with_(guards=(guard, guard.with_(statement="A different assumption.")))
    with pytest.raises(dx.ValidationError, match="surrounding whitespace"):
        original.with_(guards=(guard.with_(name=" row"),))
    with pytest.raises(dx.ValidationError, match="guards"):
        original.with_(guards=(True,))
