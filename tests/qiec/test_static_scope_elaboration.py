"""A static Reader scope compiled to ordinary nominal QIEC handlers."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EMPTY_ROW,
    REAL,
    Bind,
    Call,
    CheckContext,
    ComputationType,
    Handle,
    KernelError,
    LiteralValue,
    Local,
    Perform,
    QiecModule,
    ScopedHandlerElaboration,
    ScopedOperation,
    Var,
    discharge_module,
    elaborate_static_scope,
    exact_jvp,
    run_named,
    validate_module,
)
from quivers.qiec.numbers import host_value
from quivers.qiec.serialization import dumps, loads

_SOURCE = """
effect Reader
    read_value : Unit -> Real

instance reader : Reader
instance other : Reader

handler fixed for Reader : Real -> Real with (value : Real) [coverage=total, forwards=none, implementation=authored]
    read_value resumes 1 =>
        resume(value)

handler shifted for Reader : Real -> Real with (value : Real) [coverage=total, forwards=none, implementation=authored]
    read_value resumes 1 =>
        resume(value + 1.0)

handler copied for Reader : Real -> Real with (value : Real) [coverage=total, forwards=none, implementation=authored]
    read_value resumes omega =>
        let left <- resume(value)
        let right <- resume(value)
        return left + right

handler aborting for Reader : Real -> Real with (value : Real) [coverage=total, forwards=none, implementation=authored]
    read_value resumes 0 =>
        return value

handler nontail for Reader : Real -> Real with (value : Real) [coverage=total, forwards=none, implementation=authored]
    read_value resumes 1 =>
        let answer <- resume(value)
        return answer

define body(captured : Real) : Real !{reader} =
    let got <- perform reader.read_value()
    return got + captured

define other_body(captured : Real) : Real !{other} =
    let got <- perform other.read_value()
    return got + captured

define nested_body(captured : Real) : Real !{reader} =
    body(captured)

define demo(outer : Real, inner : Real, captured : Real) : Real !{} =
    handle reader with fixed(outer) in
        let before <- perform reader.read_value()
        let during <- body(captured)
        let after <- perform reader.read_value()
        return before * 100.0 + during * 10.0 + after
"""
_POINT = (2, 5, 7)
_DIRECTION = (1, 3, -2)


@pytest.fixture(scope="module")
def setup():
    module = Compiler(parse(_SOURCE), module_name="scoped-reader").qiec_module
    assert module is not None
    definitions = {item.name: item for item in module.computations}
    owner = definitions["demo"]
    outer = owner.body
    assert isinstance(outer, Handle) and isinstance(outer.computation, Bind)
    before = outer.computation
    assert isinstance(before.first, Perform) and isinstance(before.then, Bind)
    during = before.then
    assert isinstance(during.first, Call) and isinstance(during.then, Bind)
    after = during.then
    assert isinstance(after.first, Perform)
    source = definitions["body"]
    assert isinstance(source.body, Bind) and isinstance(source.body.first, Perform)
    other = next(
        item.entry.instance for item in module.instances if item.name == "other"
    )
    return module, definitions, outer, before, during, after, source, other


def _inputs(setup, handler_name):
    module, definitions, outer, before, _, _, source, _ = setup
    operation = ScopedOperation(
        "local", source.type, ComputationType(effects=EMPTY_ROW, result=REAL), (REAL,)
    )
    handler = next(item.id for item in module.handlers if item.name == handler_name)
    elaboration = ScopedHandlerElaboration(outer.instance, handler)
    owner = definitions["demo"]
    captured = owner.parameters[2]
    assert captured == source.parameters[0]
    context = CheckContext()
    for local in (*owner.parameters, before.binder):
        context = context.extend(local)
    return operation, elaboration, Var(local=owner.parameters[1]), captured, context


def _elaborate(setup, handler_name):
    module, _, _, _, _, _, source, _ = setup
    operation, elaboration, argument, captured, context = _inputs(setup, handler_name)
    return elaborate_static_scope(
        module,
        operation,
        elaboration,
        source.body,
        (argument,),
        captures=(captured,),
        context=context,
    )


def _replace_region(setup, term):
    module, definitions, outer, before, during, _, _, _ = setup
    shallow = before.with_(then=during.with_(first=term))
    owner = definitions["demo"].with_(body=outer.with_(computation=shallow))
    return module.with_(computations=(owner,))


def _observe(module, point):
    execution = run_named(module, "demo", point)
    requests = tuple(
        event.detail["address"]
        for event in execution.trace
        if event.event == "operation.requested"
    )
    return host_value(execution.value), requests


def test_same_static_body_changes_elaborator_and_keeps_the_outer_reader(setup):
    module, _, outer, before, _, after, source, _ = setup
    plain = _elaborate(setup, "fixed")
    shifted = _elaborate(setup, "shifted")
    assert plain.handler != shifted.handler
    assert plain.instance == shifted.instance == outer.instance
    assert plain.arguments == shifted.arguments
    assert plain.computation is shifted.computation is source.body
    expected_addresses = tuple(
        item.request.origin.dynamic_key()
        for item in (before.first, source.body.first, after.first)
    )
    assert len(set(expected_addresses)) == 3
    assert _observe(_replace_region(setup, plain), _POINT) == (
        322.0,
        expected_addresses,
    )
    assert _observe(_replace_region(setup, shifted), _POINT) == (
        332.0,
        expected_addresses,
    )
    assert _observe(_replace_region(setup, source.body), _POINT) == (
        292.0,
        expected_addresses,
    )
    # The provider never sees a second executable core or any host callback.
    assert all(handler.implementation == "authored" for handler in module.handlers)


def test_portable_scope_round_trips_and_discharges_with_the_same_value_and_jvp(setup):
    for handler, expected in (("fixed", 322), ("shifted", 332)):
        candidate = _replace_region(setup, _elaborate(setup, handler))
        text = dumps(candidate)
        restored = loads(text)
        assert isinstance(restored, QiecModule) and restored == candidate
        assert dumps(restored) == text
        assert "ScopedOperation" not in text and "ScopedHandlerElaboration" not in text
        validate_module(restored)
        original = _observe(candidate, _POINT)
        assert _observe(restored, _POINT) == original
        target = discharge_module(restored)
        assert not target.obstacles
        validate_module(target.module)
        value, trace = _observe(target.module, _POINT)
        assert value == expected
        # Discharge eliminates the handled Reader operations from the target.
        assert trace == ()
        for lowered in (restored, target.module):
            derivative = exact_jvp(lowered, "demo", _POINT, _DIRECTION)
            assert (derivative.value, derivative.tangent) == (
                Fraction(expected),
                Fraction(111),
            )


def test_move_or_copy_of_the_static_child_has_a_separating_observation(setup):
    plain = _elaborate(setup, "fixed")
    good = _observe(_replace_region(setup, plain), _POINT)
    copied_term = plain.with_(
        computation=Bind(
            first=plain.computation,
            binder=Local(name="discarded", type=REAL),
            then=plain.computation,
        )
    )
    copied = _observe(_replace_region(setup, copied_term), _POINT)
    assert copied[0] == good[0]
    assert copied[1] == (good[1][0], good[1][1], good[1][1], good[1][2])

    # Installing the inner reader outside the original region also changes
    # the before and after reads, even though the middle read is unchanged.
    _, _, outer, _, _, _, source, _ = setup
    no_local_scope = _replace_region(setup, source.body)
    owner = no_local_scope.computations[0]
    entered = owner.with_(
        body=outer.with_(computation=owner.body.computation, arguments=plain.arguments)
    )
    moved = no_local_scope.with_(computations=(entered,))
    assert _observe(moved, _POINT) == (625.0, good[1])


def test_reject_same_interface_wrong_instance_and_wrong_static_body_result(setup):
    module, definitions, _, _, _, _, source, other = setup
    operation, plan, argument, captured, context = _inputs(setup, "fixed")
    wrong_instance = replace(plan, instance=other)
    with pytest.raises(KernelError, match="declared named instance"):
        elaborate_static_scope(
            module,
            operation,
            wrong_instance,
            source.body,
            (argument,),
            captures=(captured,),
            context=context,
        )
    with pytest.raises(KernelError, match="declared named instance"):
        elaborate_static_scope(
            module,
            operation,
            plan,
            definitions["other_body"].body,
            (argument,),
            captures=(captured,),
            context=context,
        )
    with pytest.raises(KernelError, match="closed result row"):
        elaborate_static_scope(
            module,
            replace(operation, result_type=source.type),
            plan,
            source.body,
            (argument,),
            captures=(captured,),
            context=context,
        )


def test_reject_unavailable_omitted_duplicated_capture_and_different_arguments(setup):
    module, _, _, _, _, _, source, _ = setup
    operation, plan, argument, captured, context = _inputs(setup, "fixed")
    for bad, why in (
        ((Local(name="unavailable", type=REAL),), "capture .*unavailable"),
        ((), "unbound or mistyped local .*captured"),
        ((captured, captured), "local already bound"),
    ):
        with pytest.raises(KernelError, match=why):
            elaborate_static_scope(
                module,
                operation,
                plan,
                source.body,
                (argument,),
                captures=bad,
                context=context,
            )
    with pytest.raises(KernelError, match="value arguments"):
        elaborate_static_scope(
            module,
            operation,
            plan,
            source.body,
            (LiteralValue(type=REAL, value=2), argument),
            captures=(captured,),
            context=context,
        )


def test_reject_body_or_elaborator_that_can_reenter_or_skip_its_continuation(setup):
    module, definitions, _, _, _, _, _, _ = setup
    for name in ("copied", "aborting", "nontail"):
        with pytest.raises(KernelError, match="one direct tail resume"):
            _elaborate(setup, name)
    operation, plan, argument, captured, context = _inputs(setup, "fixed")
    with pytest.raises(KernelError, match="Call is outside"):
        elaborate_static_scope(
            module,
            operation,
            plan,
            definitions["nested_body"].body,
            (argument,),
            captures=(captured,),
            context=context,
        )
