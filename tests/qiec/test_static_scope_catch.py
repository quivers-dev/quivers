"""Operational controls for portable, nontransactional static catch."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    BOOL,
    REAL,
    UNIT,
    Bind,
    Call,
    Case,
    CheckContext,
    Handle,
    If,
    KernelError,
    LiteralValue,
    Local,
    NamedComputation,
    Perform,
    QiecModule,
    ScopedHandlerElaboration,
    Var,
    elaborate_static_catch,
    run_named,
    validate_module,
)
from quivers.qiec.numbers import host_value
from quivers.qiec.serialization import dumps, loads

_SOURCE = """
family Outcome : Type
    constructor Succeeded : Real -> Outcome
    constructor Failed : Real -> Outcome

effect Throw
    raise_error : Real -> Real

instance errors : Throw
instance other_errors : Throw

effect Cell
    read_value : Unit -> Real
    write_value : Real -> Unit

instance store : Cell

handler tags for Throw : Real -> Outcome [coverage=total, forwards=none, implementation=authored]
    return value =>
        return construct Succeeded(value) as Outcome
    raise_error(problem : Real) resumes 0 =>
        return construct Failed(problem) as Outcome

handler wrong_payload for Throw : Real -> Outcome [coverage=total, forwards=none, implementation=authored]
    return value =>
        return construct Succeeded(value) as Outcome
    raise_error(problem : Real) resumes 0 =>
        return construct Failed(problem + 1.0) as Outcome

handler repeated_tag for Throw : Real -> Outcome [coverage=total, forwards=none, implementation=authored]
    return value =>
        return construct Succeeded(value) as Outcome
    raise_error(problem : Real) resumes 0 =>
        return construct Succeeded(problem) as Outcome

handler resumed for Throw : Real -> Outcome [coverage=total, forwards=none, implementation=authored]
    return value =>
        return construct Succeeded(value) as Outcome
    raise_error(problem : Real) resumes 1 =>
        resume(problem)

handler outside for Throw : Real -> Real [coverage=total, forwards=none, implementation=authored]
    raise_error(problem : Real) resumes 0 =>
        return 1000.0 + problem

handler swallow for Throw : Real -> Real [coverage=total, forwards=none, implementation=authored]
    raise_error(problem : Real) resumes 0 =>
        return 0.0 - 1.0

handler state for Cell : Real -> Real with (current : Real) [coverage=total, forwards=none, implementation=authored]
    read_value resumes 1 =>
        resume(current)
    write_value(value : Real) resumes 1 =>
        resume(unit) with (value)

define attempt(next_value : Real, fail : Real, problem : Real) : Real !{store, errors} =
    perform store.write_value(next_value)
    if fail > 0.0 then
        let stopped <- perform errors.raise_error(problem)
        perform store.write_value(999.0)
        return stopped
    else
        perform store.read_value()

define other_attempt(problem : Real) : Real !{other_errors} =
    perform other_errors.raise_error(problem)

define recovery(error : Real, retry : Real) : Real !{store, errors} =
    let retained <- perform store.read_value()
    perform store.write_value(retained + 1.0)
    if retry > 0.0 then
        perform errors.raise_error(error + retained)
    else
        return retained + 2.0 * error

define under_outer(initial : Real, next_value : Real, fail : Real, retry : Real, problem : Real) : Real !{store} =
    handle errors with outside in
        let done <- attempt(next_value, fail, problem)
        return done

define demo(initial : Real, next_value : Real, fail : Real, retry : Real, problem : Real) : Real !{} =
    handle store with state(initial) in
        let value <- under_outer(initial, next_value, fail, retry, problem)
        let final_state <- perform store.read_value()
        return 100.0 * value + final_state
"""
_FAILURE = (1, 2, 1, -1, 5)
_RETHROW = (1, 2, 1, 1, 5)
_SUCCESS = (1, 2, -1, 1, 5)


@dataclass(frozen=True)
class _Fixture:
    module: QiecModule
    declaration: NamedComputation
    outside: Handle
    position: Bind
    attempt: NamedComputation
    recovery: NamedComputation

    def elaborate(self, handler="tags", **changes):
        declaration = next(
            item for item in self.module.handlers if item.name == handler
        )
        call = self.position.first
        assert isinstance(call, Call)
        options = {
            "error": self.recovery.parameters[0],
            "origin": call.origin,
            "captures": self.declaration.parameters[1:],
            "context": CheckContext(locals=self.declaration.parameters),
        }
        protected = changes.pop("protected", self.attempt.body)
        recovery = changes.pop("recovery", self.recovery.body)
        elaboration = changes.pop(
            "elaboration",
            ScopedHandlerElaboration(self.outside.instance, declaration.id),
        )
        options.update(changes)
        return elaborate_static_catch(
            self.module, elaboration, protected, recovery, **options
        )

    def install(self, term):
        body = self.outside.with_(computation=self.position.with_(first=term))
        return self.module.with_(
            computations=tuple(
                item.with_(body=body) if item == self.declaration else item
                for item in self.module.computations
            )
        )


@pytest.fixture(scope="module")
def setup():
    module = Compiler(parse(_SOURCE), module_name="static-catch").qiec_module
    assert module is not None
    items = {item.name: item for item in module.computations}
    under = items["under_outer"]
    outside = under.body
    assert isinstance(outside, Handle)
    position = outside.computation
    assert isinstance(position, Bind) and isinstance(position.first, Call)
    return _Fixture(
        module, under, outside, position, items["attempt"], items["recovery"]
    )


def _observe(module, point):
    result = run_named(module, "demo", point)
    trace = tuple(
        event.detail["address"]
        for event in result.trace
        if event.event == "operation.requested"
    )
    return host_value(result.value), trace


def _owner(module, name):
    return next(item for item in module.computations if item.name == name)


def _direct_requests(setup):
    attempt = setup.attempt.body
    recovery = setup.recovery.body
    demo = _owner(setup.module, "demo").body
    assert isinstance(attempt, Bind) and isinstance(attempt.first, Perform)
    assert isinstance(attempt.then, If)
    choice = attempt.then
    assert isinstance(choice.then, Bind) and isinstance(choice.then.first, Perform)
    assert isinstance(choice.otherwise, Perform)
    assert isinstance(recovery, Bind) and isinstance(recovery.first, Perform)
    assert isinstance(recovery.then, Bind) and isinstance(recovery.then.first, Perform)
    assert isinstance(recovery.then.then, If)
    rethrow = recovery.then.then.then
    assert isinstance(rethrow, Perform)
    assert isinstance(demo, Handle) and isinstance(demo.computation, Bind)
    finish = demo.computation.then
    assert isinstance(finish, Bind) and isinstance(finish.first, Perform)
    return (
        attempt.first,
        choice.then.first,
        choice.otherwise,
        recovery.first,
        recovery.then.first,
        rethrow,
        finish.first,
    )


def _address(term, *, in_call=True):
    site, path, shots = term.request.origin.dynamic_key()
    return (
        site,
        (*path, ("call", "under_outer#1")) if in_call else path,
        shots,
    )


def test_recovery_runs_once_with_state_from_the_interrupted_protected_body(setup):
    lowered = setup.elaborate()
    assert isinstance(lowered.first, Handle) and isinstance(lowered.then, Case)
    _, failed = lowered.then.branches
    assert lowered.first.computation is setup.attempt.body
    assert failed.body is setup.recovery.body
    module = setup.install(lowered)
    put, throw, _, read, update, _, after = _direct_requests(setup)
    assert _observe(module, _FAILURE) == (
        1203.0,
        tuple(map(_address, (put, throw, read, update)))
        + (_address(after, in_call=False),),
    )
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    assert _observe(restored, _FAILURE) == _observe(module, _FAILURE)


def test_rethrow_reaches_the_outer_handler_and_success_skips_recovery(setup):
    module = setup.install(setup.elaborate())
    put, throw, success, read, update, rethrow, after = _direct_requests(setup)
    assert _observe(module, _RETHROW) == (
        100703.0,
        tuple(map(_address, (put, throw, read, update, rethrow)))
        + (_address(after, in_call=False),),
    )
    assert _observe(module, _SUCCESS) == (
        202.0,
        tuple(map(_address, (put, success))) + (_address(after, in_call=False),),
    )


def test_independent_wrong_rollback_changes_the_observed_state_and_result(setup):
    lowered = setup.elaborate()
    put, _, _, read, _, _, _ = _direct_requests(setup)
    assert isinstance(lowered.then, Case)
    succeeded, failed = lowered.then.branches
    original = put.request
    origin = original.origin.origin
    path = origin.structural_path
    assert isinstance(path, tuple)
    reset = Perform(
        request=original.with_(
            arguments=(Var(local=setup.declaration.parameters[0]),),
            origin=original.origin.with_(
                origin=origin.with_(structural_path=(*path, "wrong-rollback")),
            ),
        )
    )
    restored = failed.with_(
        body=Bind(
            first=reset, binder=Local(name="rolled_back", type=UNIT), then=failed.body
        )
    )
    mutation = lowered.with_(then=lowered.then.with_(branches=(succeeded, restored)))
    correct = _observe(setup.install(lowered), _FAILURE)
    wrong = _observe(setup.install(mutation), _FAILURE)
    assert (correct[0], wrong[0]) == (1203.0, 1102.0)
    assert wrong[1][2:4] == (_address(reset), _address(read))


def test_independent_recovery_inside_inner_handler_swallows_the_rethrow(setup):
    lowered = setup.elaborate()
    assert isinstance(lowered.then, Case) and isinstance(lowered.first, Handle)
    success, failed = lowered.then.branches
    swallow = next(item.id for item in setup.module.handlers if item.name == "swallow")
    fault = Handle(
        instance=lowered.first.instance, handler=swallow, computation=failed.body
    )
    # Keep the ordinary declared outer row. Only the faulty arm runs.
    inside = If(
        condition=LiteralValue(type=BOOL, value=True), then=fault, otherwise=failed.body
    )
    mutation = lowered.with_(
        then=lowered.then.with_(branches=(success, failed.with_(body=inside)))
    )
    original = _observe(setup.install(lowered), _RETHROW)
    wrong = _observe(setup.install(mutation), _RETHROW)
    assert (original[0], wrong[0]) == (100703.0, -97.0)
    assert original[1] == wrong[1]


def test_requires_the_exact_nominal_exception_instance_and_pure_tags(setup):
    other = next(
        item.entry.instance
        for item in setup.module.instances
        if item.name == "other_errors"
    )
    wrong = next(item.id for item in setup.module.handlers if item.name == "tags")
    with pytest.raises(KernelError, match="declared named exception instance"):
        setup.elaborate(elaboration=ScopedHandlerElaboration(other, wrong))
    with pytest.raises(KernelError, match="declared named exception instance"):
        setup.elaborate(protected=_owner(setup.module, "other_attempt").body)
    with pytest.raises(KernelError, match="unchanged payload"):
        setup.elaborate("wrong_payload")
    with pytest.raises(KernelError, match="different.*constructors"):
        setup.elaborate("repeated_tag")
    with pytest.raises(KernelError, match="grade-zero"):
        setup.elaborate("resumed")
    with pytest.raises(KernelError, match="scalar recovery binder"):
        setup.elaborate(error=Local(name="problem", type=BOOL))


def test_catch_does_not_capture_unlisted_values_and_avoids_internal_name_collision(
    setup,
):
    with pytest.raises(KernelError, match="unbound or mistyped local"):
        setup.elaborate(captures=())
    with pytest.raises(KernelError, match="unavailable"):
        setup.elaborate(captures=(Local(name="missing", type=REAL),))

    call = setup.position.first
    assert isinstance(call, Call)
    site = str(call.origin.site_id())
    colliding = Local(name=f"_qvr_caught_{site[-24:]}", type=REAL)
    existing = CheckContext(locals=(*setup.declaration.parameters, colliding))
    term = setup.elaborate(context=existing)
    assert term.binder.name != colliding.name
    assert isinstance(term.then, Case)
    assert term.then.branches[0].scope != term.then.branches[1].scope
    validate_module(setup.install(term))
