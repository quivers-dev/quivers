"""Nominal effect-law obligation records and their checked links."""

from hashlib import sha256
import json
from typing import Any

import didactic.api as dx
import pytest

from quivers.core.laws import PLUS_ASSOCIATIVE, ClaimSubject, LawClaim, Proved
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    BOOL,
    QIEC_WIRE_FORMAT,
    Condition,
    EffectDef,
    EffectEquation,
    EffectId,
    EffectLawBundle,
    EffectLawObligation,
    EffectRef,
    HandlerId,
    OperationDef,
    OperationId,
    QiecModule,
    SerializationError,
    check_finite_effect_law,
    dumps,
    from_data,
    loads,
    to_data,
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

define truth() : Bool !{} =
    return true
"""


@pytest.fixture(scope="module")
def bundle() -> EffectLawBundle:
    module = Compiler(parse(_SOURCE), module_name="effect-law-records").qiec_module
    assert module is not None
    row = next(item.entry for item in module.instances if item.name == "reader")
    interface = next(item for item in module.effects if item.ref.id == row.effect.id)
    left, right, _ = module.computations
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


def test_bundle_roundtrip_retains_equations_and_verifies_freshly(
    bundle: EffectLawBundle,
) -> None:
    encoded = dumps(bundle)
    restored = loads(encoded)
    assert isinstance(restored, EffectLawBundle)
    assert restored == bundle
    assert dumps(restored) == encoded
    assert json.loads(encoded)["$schema"] == QIEC_WIRE_FORMAT
    assert dumps(restored.module) == dumps(bundle.module)
    obligation = restored.obligations[0]
    assert loads(dumps(obligation)) == obligation
    assert loads(dumps(obligation.equation)) == obligation.equation
    actual = check_finite_effect_law(
        restored.module, obligation.equation, obligation.handler
    )
    assert actual.certified
    assert actual.result.checked == 2


def test_declared_hypotheses_cannot_supply_a_successful_decision(
    bundle: EffectLawBundle,
) -> None:
    original = bundle.obligations[0]
    wrong = original.with_(
        equation=original.equation.with_(right=bundle.module.computations[2].id),
        guards=(Condition(name="asserts-equality", statement="The answers agree."),),
    )
    restored = loads(dumps(bundle.with_(obligations=(wrong,))))
    assert isinstance(restored, EffectLawBundle)
    spec = restored.obligations[0]
    result = check_finite_effect_law(restored.module, spec.equation, spec.handler)
    assert not result.certified
    witness = result.result.counterexample
    assert witness is not None
    assert (witness.arguments, witness.left, witness.right) == ((False,), False, True)


def test_existing_v2_effect_module_preserves_its_canonical_bytes() -> None:
    identity = EffectId.derive("legacy", "Read")
    effect = EffectDef(
        ref=EffectRef(id=identity, name="Read"),
        operations=(
            OperationDef(
                id=OperationId.derive(str(identity), "read"),
                name="read",
                telescope=(),
                arguments=(),
                result_type=BOOL,
            ),
        ),
        telescope=(),
    )
    module = QiecModule(module="legacy", source_protocol="legacy/v1", effects=(effect,))
    encoded = dumps(module)
    # Recorded with the pre-obligation QIEC v2 encoder.
    assert sha256(encoded.encode()).hexdigest() == (
        "b4dd77bcb3407eade914537a522b61d2a27dfc5a01ed9bc3987ad5736e8f3869"
    )
    assert loads(encoded) == module


def _encoded_field(node: dict[str, Any], name: str) -> dict[str, Any]:
    return next(entry for entry in node["fields"] if entry["name"] == name)


def test_wire_rejects_duplicate_guard_names_and_unknown_guard_shapes(
    bundle: EffectLawBundle,
) -> None:
    data = json.loads(dumps(bundle))
    obligations = _encoded_field(data["root"], "obligations")["value"]["$tuple"]
    guards = _encoded_field(obligations[0], "guards")["value"]["$tuple"]
    guards.append(guards[0])
    with pytest.raises(SerializationError, match="duplicate effect law guard name"):
        from_data(data)
    guards.pop()
    original = guards[0]
    guards[0] = {"$type": "effect_law_schema.FutureGuard", "fields": []}
    with pytest.raises(SerializationError, match="unknown QIEC node tag"):
        from_data(data)
    guards[0] = True
    with pytest.raises(SerializationError, match=r"EffectLawObligation\.guards"):
        from_data(data)
    guards[0] = original
    _encoded_field(original, "name")["value"] = " "
    with pytest.raises(SerializationError, match="condition's name"):
        from_data(data)


def test_wire_rejects_forged_success_and_unknown_methods(
    bundle: EffectLawBundle,
) -> None:
    data = json.loads(dumps(bundle))
    obligations = _encoded_field(data["root"], "obligations")["value"]["$tuple"]
    obligation = obligations[0]
    obligation["fields"].append({"name": "certified", "value": True})
    with pytest.raises(SerializationError, match=r"fields for .*EffectLawObligation"):
        from_data(data)
    obligation["fields"].pop()
    _encoded_field(obligation, "method")["value"] = "proved"
    with pytest.raises(SerializationError, match=r"EffectLawObligation\.method"):
        from_data(data)
    _encoded_field(obligation, "method")["value"] = "finite"
    forged = LawClaim(
        law=PLUS_ASSOCIATIVE,
        subject=ClaimSubject(
            kind="computation", name=str(bundle.obligations[0].handler)
        ),
        evidence=Proved(argument="claimed proof"),
    )
    obligations[0] = to_data(forged)["root"]
    with pytest.raises(SerializationError, match=r"EffectLawBundle\.obligations"):
        from_data(data)


def test_wire_rejects_duplicate_and_mismatched_obligations(
    bundle: EffectLawBundle,
) -> None:
    data = json.loads(dumps(bundle))
    obligations = _encoded_field(data["root"], "obligations")["value"]["$tuple"]
    obligations.append(obligations[0])
    with pytest.raises(SerializationError, match="duplicate effect law obligation"):
        from_data(data)
    obligations.pop()
    foreign = HandlerId.derive("unrelated", "fixed")
    _encoded_field(obligations[0], "handler")["value"] = to_data(foreign)["root"]
    with pytest.raises(SerializationError, match="mismatched module declaration"):
        from_data(data)
