"""Static discharge of authored handlers.

A discharged computation installs no handler, has the type of the original, and
has its value at every argument: the reference machine runs both. The random
programs request under handlers of every grade in a random nesting order, with
parameters, with clauses that resume in and out of tail position, and with
requests under conditionals and inside callees.
"""

from __future__ import annotations

from itertools import product

import pytest
import torch

import quivers.qiec.discharge as discharge
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    RANDOM_SAMPLE,
    SCORE_ADD,
    CheckContext,
    DISCHARGE_STATEMENTS,
    Bind,
    Call,
    Case,
    Computation,
    DischargeObstacle,
    DischargedModule,
    ExecutionFailure,
    Handle,
    If,
    InstanceAddress,
    NamedComputation,
    NewInstance,
    Perform,
    QiecModule,
    Resume,
    RuntimeConfiguration,
    RuntimeSelection,
    WithTraceAddresses,
    discharge_computation,
    discharge_module,
    infer_computation,
    run_named,
    validate_module,
)
from quivers.qiec.serialization import dumps, loads
from tests._handler_programs import HANDLERS, random_body

_POINTS = ((0.3137, -0.7171), (1.2113, 0.4092), (-0.9049, 0.8117))


def _module(source: str) -> QiecModule:
    """A source's checked module."""
    compiler = Compiler(parse(source), module_name="discharge", file_path="d.qvr")
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _computation(module: QiecModule, name: str) -> NamedComputation:
    """A module's computation of a name."""
    return next(item for item in module.computations if item.name == name)


def _terms(body: Computation) -> list[Computation]:
    """Every computation term of a body."""
    found: list[Computation] = []
    pending: list[Computation] = [body]
    while pending:
        term = pending.pop()
        found.append(term)
        if isinstance(term, Bind):
            pending += [term.first, term.then]
        elif isinstance(term, If):
            pending += [term.then, term.otherwise]
        elif isinstance(term, Case):
            pending += [branch.body for branch in term.branches]
        elif isinstance(term, Handle):
            pending.append(term.computation)
        elif isinstance(term, NewInstance | WithTraceAddresses):
            pending.append(term.body)
    return found


def _discharged(module: QiecModule, name: str) -> NamedComputation:
    """A computation discharged, checked to have its declared type."""
    result = discharge_module(module)
    assert result.obstacle(name) is None
    registry = validate_module(module)
    computation = _computation(result.module, name)
    assert not any(
        isinstance(term, Handle | Resume | NewInstance)
        for term in _terms(computation.body)
    )
    context = CheckContext()
    for parameter in computation.parameters:
        context = context.extend(parameter)
    actual = infer_computation(computation.body, registry, context)
    assert actual.result == computation.type.result
    declared = {entry.instance for entry in computation.type.effects.entries}
    assert {entry.instance for entry in actual.effects.entries} <= declared
    statements = [
        term for term in _terms(computation.body) if not isinstance(term, Bind)
    ]
    assert len(statements) <= DISCHARGE_STATEMENTS
    return computation


def _agree(module: QiecModule, name: str, arguments: tuple[object, ...]) -> object:
    """A computation's value, the same before and after discharge."""
    _discharged(module, name)
    original = run_named(module, name, arguments)
    rewritten = run_named(discharge_module(module).module, name, arguments)
    torch.testing.assert_close(rewritten.value, original.value, rtol=1e-9, atol=1e-12)
    assert not any(event.event.startswith("handler") for event in rewritten.trace)
    assert any(event.event == "handler.entered" for event in original.trace)
    return original.value


_ADDRESS_SOURCE = """
effect Fork
    split : Real -> Real
effect Relay
    pass : Real -> Real

instance branch : Fork
instance outside_branch : Fork
instance relay : Relay
instance random : Random
instance score : Score

handler twice for Fork : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    split(v : Real) resumes omega =>
        let first <- resume(v)
        let second <- resume(v)
        return first + second

handler guarded_shots for Fork : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    split(v : Real) resumes omega =>
        if v > 0.0 then
            let first <- resume(v)
            resume(v)
        else
            resume(v)

handler logged_shots for Fork : Real -> Real [coverage=total, forwards=none, introduces=!{score}, implementation=authored]
    return x =>
        return x
    split(v : Real) resumes omega =>
        let first <- resume(v)
        perform score.add(weight(first))
        resume(v)

handler identity for Relay : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    pass(v : Real) resumes 1 =>
        resume(v)

handler draw for Random : Real -> Real [coverage=total, implementation=foreign]
    sample[A : Type] resumes 1

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define flat(x : Real) : Real !{random} =
    handle branch with twice in
        let first <- perform branch.split(x)
        let second <- perform branch.split(first)
        let z <- perform random.sample[Real](site("z"), Normal(second, 1.0))
        return z

define branch_shots(x : Real) : Real !{random} =
    handle branch with guarded_shots in
        let v <- perform branch.split(x)
        let z <- perform random.sample[Real](site("z"), Normal(v, 1.0))
        return z

define logged(x : Real) : Real !{random, score} =
    handle branch with logged_shots in
        let v <- perform branch.split(x)
        let z <- perform random.sample[Real](site("z"), Normal(v, 1.0))
        return z

define relay_score(x : Real) : Real !{random, score} =
    handle relay with identity in
        let z <- perform relay.pass(x)
        let w <- perform random.sample[Real](site("w"), Normal(z, 1.0))
        perform score.add(weight(w))
        return w

define mixed_omega(x : Real) : Real !{outside_branch, random} =
    handle branch with guarded_shots in
        let first <- perform branch.split(x)
        let second <- perform outside_branch.split(first)
        let z <- perform random.sample[Real](site("mixed"), Normal(second, 1.0))
        return z

define mixed_linear(x : Real) : Real !{relay, random} =
    handle branch with guarded_shots in
        let first <- perform branch.split(x)
        let second <- perform relay.pass(first)
        let z <- perform random.sample[Real](site("mixed"), Normal(second, 1.0))
        return z

define flat_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle random with draw in
            let y <- flat(x)
            perform score.add(weight(y))
            return y

define branch_run(x : Real) : Real !{} =
    handle random with draw in
        branch_shots(x)

define logged_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle random with draw in
            logged(x)

define linear_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle random with draw in
            relay_score(x)

define nested_run(x : Real) : Real !{} =
    handle random with draw in
        handle branch with twice in
            let v <- perform branch.split(x)
            flat(v)

define mixed_omega_run(x : Real) : Real !{} =
    handle random with draw in
        handle outside_branch with twice in
            mixed_omega(x)

define mixed_linear_run(x : Real) : Real !{} =
    handle random with draw in
        handle relay with identity in
            mixed_linear(x)
"""


@pytest.fixture(scope="module")
def address_module() -> tuple[QiecModule, DischargedModule]:
    """The checked module and its discharged copy.

    The foreign handlers in the entry points keep their callers' scopes live
    while the authored handlers in the callees are discharged.
    """
    module = _module(_ADDRESS_SOURCE)
    discharged = discharge_module(module)
    for name in ("flat", "branch_shots", "logged", "relay_score", "mixed_linear"):
        assert discharged.obstacle(name) is None
    return module, discharged


def _address_site(module: QiecModule, name: str, operation: str) -> str:
    return next(
        str(term.request.origin.static_site)
        for term in _terms(_computation(module, name).body)
        if isinstance(term, Perform) and str(term.request.operation) == operation
    )


def _request_addresses(
    module: QiecModule, name: str, value: float
) -> tuple[object, ...]:
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core",
                options={
                    "handlers": {
                        "draw": {"kind": "draw"},
                        "accumulate": {"kind": "score"},
                    }
                },
            ),
        )
    )
    result = run_named(module, name, (value,), runtime=runtime)
    return tuple(
        event.detail["address"]
        for event in result.trace
        if event.event == "operation.requested"
        and event.detail["operation"] in (str(RANDOM_SAMPLE), str(SCORE_ADD))
    )


def test_discharge_keeps_equal_valued_shots_and_restores_the_outer_path(
    address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = address_module
    draw = _address_site(module, "flat", str(RANDOM_SAMPLE))
    score = _address_site(module, "flat_run", str(SCORE_ADD))
    expected = (
        *((draw, (("call", "flat#1"),), path) for path in product(range(2), repeat=2)),
        (score, (), ()),
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "flat_run", 0.0) == expected


@pytest.mark.parametrize(("value", "count"), ((-1.0, 1), (1.0, 2)))
def test_a_symbolic_branch_keeps_each_clause_invocation_shot_count(
    address_module: tuple[QiecModule, DischargedModule], value: float, count: int
) -> None:
    module, discharged = address_module
    draw = _address_site(module, "branch_shots", str(RANDOM_SAMPLE))
    expected = tuple(
        (draw, (("call", "branch_shots#1"),), (index,)) for index in range(count)
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "branch_run", value) == expected


def test_a_request_between_shots_uses_the_surrounding_path(
    address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = address_module
    draw = _address_site(module, "logged", str(RANDOM_SAMPLE))
    handler = next(
        handler for handler in module.handlers if handler.name == "logged_shots"
    )
    body = handler.clauses[0].body
    assert body is not None
    score = next(
        str(term.request.origin.static_site)
        for term in _terms(body)
        if isinstance(term, Perform) and term.request.operation == SCORE_ADD
    )
    expected = (
        (draw, (("call", "logged#1"),), (0,)),
        (score, (("call", "logged#1"),), ()),
        (draw, (("call", "logged#1"),), (1,)),
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "logged_run", 0.0) == expected


def test_linear_handlers_leave_no_shot_labels(
    address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = address_module
    draw = _address_site(module, "relay_score", str(RANDOM_SAMPLE))
    score = _address_site(module, "relay_score", str(SCORE_ADD))
    expected = (
        (draw, (("call", "relay_score#1"),), ()),
        (score, (("call", "relay_score#1"),), ()),
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "linear_run", 0.0) == expected


def test_a_callers_live_shot_precedes_discharged_shots_in_its_callee(
    address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = address_module
    draw = _address_site(module, "flat", str(RANDOM_SAMPLE))
    expected = tuple(
        (draw, (("call", f"flat#{outer + 1}"),), (outer, *inner))
        for outer in range(2)
        for inner in product(range(2), repeat=2)
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "nested_run", 0.0) == expected


def test_an_interleaved_runtime_shot_keeps_the_reference_computation(
    address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = address_module
    obstacle = discharged.obstacle("mixed_omega")
    assert obstacle is not None and obstacle.kind == "address"
    assert "Fork.split" in obstacle.reason
    assert _computation(discharged.module, "mixed_omega") is _computation(
        module, "mixed_omega"
    )
    draw = _address_site(module, "mixed_omega", str(RANDOM_SAMPLE))
    expected = tuple(
        (draw, (("call", "mixed_omega#1"),), (0, index)) for index in range(2)
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "mixed_omega_run", -1.0) == expected


def test_an_interleaved_linear_runtime_handler_allows_discharge(
    address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = address_module
    assert not any(
        isinstance(term, Handle)
        for term in _terms(_computation(discharged.module, "mixed_linear").body)
    )
    draw = _address_site(module, "mixed_linear", str(RANDOM_SAMPLE))
    expected = ((draw, (("call", "mixed_linear#1"),), (0,)),)
    for checked in (module, discharged.module):
        assert _request_addresses(checked, "mixed_linear_run", -1.0) == expected


_CLAUSE_SOURCE = """
effect Tap
    echo : Real -> Real

instance tap : Tap

handler recording_relay for Relay : Real -> Real [coverage=total, forwards=none, introduces=!{score}, implementation=authored]
    return x =>
        return x
    pass(v : Real) resumes 1 =>
        perform score.add(weight(v))
        resume(v)

handler nesting_relay for Relay : Real -> Real [coverage=total, forwards=none, introduces=!{score, tap}, implementation=authored]
    return x =>
        return x
    pass(v : Real) resumes 1 =>
        let forwarded <- perform tap.echo(v)
        perform score.add(weight(forwarded))
        resume(forwarded)

handler recording_tap for Tap : Real -> Real [coverage=total, forwards=none, introduces=!{score}, implementation=authored]
    return x =>
        return x
    echo(v : Real) resumes 1 =>
        perform score.add(weight(v))
        resume(v)

handler delegating_relay for Relay : Real -> Real [coverage=total, forwards=none, introduces=!{score}, implementation=authored]
    return x =>
        return x
    pass(v : Real) resumes 1 =>
        let result <- report_twice(v)
        resume(result)

define report_twice(x : Real) : Real !{score} =
    handle relay with recording_relay in
        let first <- perform relay.pass(x)
        let second <- perform relay.pass(first)
        perform score.add(weight(second))
        return second

define report_nested(x : Real) : Real !{score} =
    handle tap with recording_tap in
        handle relay with nesting_relay in
            let first <- perform relay.pass(x)
            let second <- perform relay.pass(first)
            perform score.add(weight(second))
            return second

define report_through_call(x : Real) : Real !{score} =
    handle relay with delegating_relay in
        let first <- perform relay.pass(x)
        let second <- perform relay.pass(first)
        return second

define report_twice_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let y <- report_twice(x)
        perform score.add(weight(y))
        return y

define report_nested_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let y <- report_nested(x)
        perform score.add(weight(y))
        return y

define report_through_call_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        report_through_call(x)
"""


@pytest.fixture(scope="module")
def clause_address_module() -> tuple[QiecModule, DischargedModule]:
    module = _module(_ADDRESS_SOURCE + _CLAUSE_SOURCE)
    return module, discharge_module(module)


def _effect_sites(body: Computation, effect: str) -> tuple[str, ...]:
    requests = [
        term.request
        for term in _terms(body)
        if isinstance(term, Perform) and term.request.effect.name == effect
    ]
    requests.sort(key=lambda request: request.origin.origin.line or 0)
    return tuple(str(request.origin.static_site) for request in requests)


def _effect_clause_body(module: QiecModule, handler: str) -> Computation:
    body = (
        next(item for item in module.handlers if item.name == handler).clauses[0].body
    )
    assert body is not None
    return body


@pytest.mark.parametrize("nested", (False, True))
def test_linear_clauses_address_repeated_effects_by_trigger_site(
    clause_address_module: tuple[QiecModule, DischargedModule], nested: bool
) -> None:
    module, discharged = clause_address_module
    name = "report_nested" if nested else "report_twice"
    assert discharged.obstacle(name) is None
    original = _computation(module, name)
    rewritten = _computation(discharged.module, name)
    assert not any(isinstance(item, Handle) for item in _terms(rewritten.body))
    triggers = _effect_sites(original.body, "Relay")
    assert len(triggers) == 2
    handler = "nesting_relay" if nested else "recording_relay"
    clause = _effect_clause_body(module, handler)
    (clause_score,) = _effect_sites(clause, "Score")
    (after_resume,) = _effect_sites(original.body, "Score")
    (outside,) = _effect_sites(_computation(module, name + "_run").body, "Score")
    call = (("call", f"{name}#1"),)
    expected = []
    for trigger in triggers:
        path = (*call, ("handler-clause", trigger))
        if nested:
            (relay,) = _effect_sites(clause, "Tap")
            (relay_score,) = _effect_sites(
                _effect_clause_body(module, "recording_tap"), "Score"
            )
            expected.append((relay_score, (*path, ("handler-clause", relay)), ()))
        expected.append((clause_score, path, ()))
    expected.extend(((after_resume, call, ()), (outside, (), ())))
    round_trip = loads(dumps(discharged.module))
    assert isinstance(round_trip, QiecModule)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, name + "_run", 1.0) == tuple(expected)


def test_an_effectful_call_inside_a_linear_clause_restores_its_path(
    clause_address_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = clause_address_module
    name = "report_through_call"
    assert discharged.obstacle(name) is None
    original = _computation(module, name)
    rewritten = _computation(discharged.module, name)
    assert not any(isinstance(term, Handle) for term in _terms(rewritten.body))
    assert discharged.obstacle("report_twice") is None
    callee = _computation(module, "report_twice").body
    nested = _effect_sites(callee, "Relay")
    (after_resume,) = _effect_sites(callee, "Score")
    (score,) = _effect_sites(_effect_clause_body(module, "recording_relay"), "Score")
    expected = []
    for index, trigger in enumerate(_effect_sites(original.body, "Relay"), start=2):
        path = (
            ("call", f"{name}#1"),
            ("call", f"report_twice#{index}"),
            ("handler-clause", trigger),
        )
        expected.extend(
            (score, (*path, ("handler-clause", site)), ()) for site in nested
        )
        expected.append((after_resume, path, ()))
    round_trip = loads(dumps(discharged.module))
    assert isinstance(round_trip, QiecModule)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, name + "_run", 1.0) == tuple(expected)


_CAPTURED_CALL_SOURCE = """
define relay_leaf(x : Real) : Real !{relay} =
    let result <- perform relay.pass(x)
    return result

define relay_chain(x : Real) : Real !{relay} =
    let result <- relay_leaf(x)
    return result

define relay_tail(x : Real) : Real !{relay} =
    if x > 0.0 then
        relay_leaf(x)
    else
        relay_leaf(0.0 - x)

define relay_cycle(n : Int, x : Real) : Real !{relay, score} =
    if n > 0 then
        relay_cycle(n - 1, x)
    else
        let result <- perform relay.pass(x)
        perform score.add(weight(result))
        return result

define relay_lexical_cycle(n : Int, x : Real) : Real !{relay} =
    if n > 0 then
        relay_lexical_cycle(n - 1, x)
    else
        let first <- relay_leaf(x)
        let second <- relay_leaf(first)
        return second

define report_captured(x : Real) : Real !{score} =
    handle relay with recording_relay in
        let first <- relay_chain(x)
        let second <- relay_chain(first)
        perform score.add(weight(second))
        return second

define report_tail(x : Real) : Real !{score} =
    handle relay with recording_relay in
        let first <- relay_tail(x)
        let second <- relay_tail(first)
        perform score.add(weight(second))
        return second

define report_pure_tail(x : Real) : Real !{score} =
    handle relay with identity in
        let result <- relay_tail(x)
        perform score.add(weight(result))
        return result

define report_cycle(x : Real) : Real !{score} =
    handle relay with recording_relay in
        let first <- relay_cycle(64, x)
        let second <- relay_cycle(64, first)
        perform score.add(weight(second))
        return second

define report_lexical_cycle(x : Real) : Real !{score} =
    handle relay with recording_relay in
        relay_lexical_cycle(3, x)

define report_captured_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        report_captured(x)

define report_tail_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        report_tail(x)

define report_pure_tail_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        report_pure_tail(x)

define report_cycle_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let result <- report_cycle(x)
        perform score.add(weight(result))
        return result

define report_lexical_cycle_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        report_lexical_cycle(x)
"""


@pytest.fixture(scope="module")
def captured_call_module() -> tuple[QiecModule, DischargedModule]:
    module = _module(_ADDRESS_SOURCE + _CLAUSE_SOURCE + _CAPTURED_CALL_SOURCE)
    return module, discharge_module(module)


@pytest.mark.parametrize(
    ("name", "helper"),
    (("report_captured", "relay_chain"), ("report_tail", "relay_tail")),
)
def test_captured_call_words_distinguish_one_lexical_operation(
    captured_call_module: tuple[QiecModule, DischargedModule],
    name: str,
    helper: str,
) -> None:
    module, discharged = captured_call_module
    assert discharged.obstacle(name) is None
    original = _computation(module, name).body
    rewritten = _computation(discharged.module, name).body
    assert not any(isinstance(item, Handle | Call) for item in _terms(rewritten))
    entries = sorted(
        (term for term in _terms(original) if isinstance(term, Call)),
        key=lambda term: term.origin.line or 0,
    )
    assert len(entries) == 2
    leaf_call = min(
        (
            term
            for term in _terms(_computation(module, helper).body)
            if isinstance(term, Call)
        ),
        key=lambda term: term.origin.line or 0,
    )
    (trigger,) = _effect_sites(_computation(module, "relay_leaf").body, "Relay")
    (score,) = _effect_sites(_effect_clause_body(module, "recording_relay"), "Score")
    (outside,) = _effect_sites(original, "Score")
    call = (("call", f"{name}#1"),)
    expected = (
        *(
            (
                score,
                (
                    *call,
                    (
                        "handler-clause",
                        "/".join(
                            (
                                trigger,
                                str(entry.origin.site_id()),
                                str(leaf_call.origin.site_id()),
                            )
                        ),
                    ),
                ),
                (),
            )
            for entry in entries
        ),
        (outside, call, ()),
    )
    assert len(set(expected)) == 3
    round_trip = loads(dumps(discharged.module))
    assert isinstance(round_trip, QiecModule)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, name + "_run", 1.0) == expected


def test_a_pure_clause_does_not_need_an_elided_callers_address(
    captured_call_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = captured_call_module
    name = "report_pure_tail"
    assert discharged.obstacle(name) is None
    rewritten = _computation(discharged.module, name).body
    assert not any(isinstance(item, Handle | Call) for item in _terms(rewritten))
    (score,) = _effect_sites(_computation(module, name).body, "Score")
    expected = ((score, (("call", f"{name}#1"),), ()),)
    for checked in (module, discharged.module):
        assert _request_addresses(checked, name + "_run", 1.0) == expected


def test_recurrent_tail_calls_keep_their_own_clause_activation(
    captured_call_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = captured_call_module
    name = "report_cycle"
    obstacle = discharged.obstacle(name)
    assert obstacle is not None and obstacle.kind == "recursion"
    recursive = _computation(module, "relay_cycle").body
    (trigger,) = _effect_sites(recursive, "Relay")
    (score,) = _effect_sites(_effect_clause_body(module, "recording_relay"), "Score")
    (resumed,) = _effect_sites(recursive, "Score")
    (outside,) = _effect_sites(_computation(module, name).body, "Score")
    (after_call,) = _effect_sites(_computation(module, name + "_run").body, "Score")
    parent = (("call", f"{name}#1"),)
    expected = []
    # Each loop enters once at 64 and once at each smaller value down to 0.
    for active in (1 + 65, 1 + 2 * 65):
        clause = ("handler-clause", f"{trigger}/runtime-call:{active}")
        expected.append((score, (*parent, clause), ()))
        expected.append((resumed, (*parent, ("call", f"relay_cycle#{active}")), ()))
    expected.extend(((outside, parent, ()), (after_call, (), ())))
    assert len(set(expected)) == len(expected)
    round_trip = loads(dumps(discharged.module))
    assert isinstance(round_trip, QiecModule)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, name + "_run", 1.0) == tuple(expected)


def test_acyclic_call_sites_stay_ordered_after_a_recurrent_frame(
    captured_call_module: tuple[QiecModule, DischargedModule],
) -> None:
    module, discharged = captured_call_module
    name = "report_lexical_cycle"
    obstacle = discharged.obstacle(name)
    assert obstacle is not None and obstacle.kind == "recursion"
    (trigger,) = _effect_sites(_computation(module, "relay_leaf").body, "Relay")
    (score,) = _effect_sites(_effect_clause_body(module, "recording_relay"), "Score")
    calls = sorted(
        (
            term
            for term in _terms(_computation(module, "relay_lexical_cycle").body)
            if isinstance(term, Call) and term.name == "relay_leaf"
        ),
        key=lambda term: term.origin.line or 0,
    )
    assert len(calls) == 2
    # The wrapper has ordinal 1, then four recursive entries reach zero.
    expected = tuple(
        (
            score,
            (
                ("call", f"{name}#1"),
                ("handler-clause", f"{trigger}/runtime-call:5/{call.origin.site_id()}"),
            ),
            (),
        )
        for call in calls
    )
    for checked in (module, discharged.module):
        assert _request_addresses(checked, name + "_run", 1.0) == expected


_INLINED_ADDRESS_SOURCE = """
define scoped_leaf(x : Real) : Real !{relay, score} =
    perform score.add(weight(x))
    let y <- perform relay.pass(x)
    perform score.add(weight(y))
    return y

define scoped_tail(x : Real) : Real !{relay, score} =
    scoped_leaf(x)

define scoped_parent(x : Real) : Real !{score} =
    handle relay with recording_relay in
        let y <- scoped_tail(x)
        perform score.add(weight(y))
        return y

define scoped_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        scoped_parent(x)

define scoped_draw(x : Real) : Real !{branch, random} =
    let y <- perform branch.split(x)
    let z <- perform random.sample[Real](site("inside"), Normal(y, 1.0))
    return z

define scoped_shots(x : Real) : Real !{random} =
    handle branch with twice in
        scoped_draw(x)

define scoped_shots_run(x : Real) : Real !{} =
    handle random with draw in
        scoped_shots(x)

define ordinal_leaf(x : Real) : Real !{relay} =
    let y <- perform relay.pass(x)
    return y

define ordinal_pure(x : Real) : Real !{} =
    handle relay with identity in
        ordinal_leaf(x)

define ordinal_score(x : Real) : Real !{score} =
    perform score.add(weight(x))
    return x

define ordinal_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let value <- ordinal_pure(x)
        ordinal_score(value)

define scoped_bound_call(x : Real) : Real !{relay, score} =
    let y <- perform relay.pass(x)
    let z <- ordinal_score(y)
    return z

define scoped_tail_call(x : Real) : Real !{relay, score} =
    let y <- perform relay.pass(x)
    ordinal_score(y)

define scoped_bound_parent(x : Real) : Real !{score} =
    handle relay with identity in
        scoped_bound_call(x)

define scoped_tail_parent(x : Real) : Real !{score} =
    handle relay with identity in
        scoped_tail_call(x)

define scoped_bound_parent_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        scoped_bound_parent(x)

define scoped_tail_parent_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        scoped_tail_parent(x)

define scoped_tap_leaf(x : Real) : Real !{tap} =
    let value <- perform tap.echo(x)
    return value

define scoped_tap_tail(x : Real) : Real !{relay, tap} =
    let y <- perform relay.pass(x)
    scoped_tap_leaf(y)

define scoped_tap_parent(x : Real) : Real !{tap} =
    handle relay with identity in
        scoped_tap_tail(x)

define scoped_tap_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle tap with recording_tap in
            scoped_tap_parent(x)

define local_subject(x : Real) : Real !{score} =
    with instance local : Relay in
        handle local with identity in
            let y <- perform local.pass(x)
            perform score.add(weight(y))
            return y

define local_pure(x : Real) : Real !{} =
    with instance local : Relay in
        handle local with identity in
            let y <- perform local.pass(x)
            return y

define repeated_local_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let first <- local_subject(x)
        let second <- local_subject(first)
        perform score.add(weight(second))
        return second

define ordinal_local_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let y <- local_pure(x)
        local_subject(y)

define mixed_local_parent(x : Real) : Real !{score} =
    with instance local : Relay in
        handle relay with recording_relay in
            scoped_leaf(x)

define mixed_local_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        mixed_local_parent(x)

define retained_local_parent(x : Real) : Real !{score} =
    with instance local : Relay in
        ordinal_score(x)

define retained_local_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        retained_local_parent(x)

define residual_draw(x : Real) : Real !{random} =
    let value <- perform random.sample[Real](site("residual"), Normal(x, 1.0))
    return value

define residual_shot_subject(x : Real) : Real !{random} =
    handle branch with twice in
        let y <- perform branch.split(x)
        let inside <- residual_draw(y)
        let after <- perform random.sample[Real](site("after-call"), Normal(inside, 1.0))
        return after

define residual_shot_run(x : Real) : Real !{} =
    handle random with draw in
        residual_shot_subject(x)

define residual_nested_run(x : Real) : Real !{} =
    handle random with draw in
        handle outside_branch with twice in
            let y <- perform outside_branch.split(x)
            residual_shot_subject(y)

define residual_omega_leaf(x : Real) : Real !{outside_branch, random} =
    let y <- perform outside_branch.split(x)
    let z <- perform random.sample[Real](site("inner"), Normal(y, 1.0))
    return z

define residual_omega_subject(x : Real) : Real !{outside_branch, random} =
    handle branch with guarded_shots in
        let y <- perform branch.split(x)
        residual_omega_leaf(y)

define residual_omega_call_run(x : Real) : Real !{} =
    handle random with draw in
        handle outside_branch with twice in
            residual_omega_subject(x)
"""


@pytest.fixture(scope="module")
def inlined_address_module() -> tuple[QiecModule, DischargedModule, QiecModule]:
    module = _module(_ADDRESS_SOURCE + _CLAUSE_SOURCE + _INLINED_ADDRESS_SOURCE)
    discharged = discharge_module(module)
    round_trip = loads(dumps(discharged.module))
    assert isinstance(round_trip, QiecModule)
    return module, discharged, round_trip


def test_inlined_tail_call_address_survives_capture_and_resume(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("scoped_parent") is None
    parent = _computation(module, "scoped_parent").body
    tail = _computation(module, "scoped_tail").body
    leaf = _computation(module, "scoped_leaf").body
    outer = (("call", "scoped_parent#1"),)
    active = (*outer, ("call", "scoped_leaf#3"))
    (request,) = _effect_sites(leaf, "Relay")
    (before, after) = _effect_sites(leaf, "Score")
    (outside,) = _effect_sites(parent, "Score")
    (clause,) = _effect_sites(_effect_clause_body(module, "recording_relay"), "Score")
    first = next(
        term.origin.site_id() for term in _terms(parent) if isinstance(term, Call)
    )
    second = next(
        term.origin.site_id() for term in _terms(tail) if isinstance(term, Call)
    )
    clause_path = (*outer, ("handler-clause", f"{request}/{first}/{second}"))
    expected = (
        (before, active, ()),
        (clause, clause_path, ()),
        (after, active, ()),
        (outside, outer, ()),
    )
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "scoped_run", 1.0) == expected


def test_inlined_call_keeps_one_activation_in_both_shots(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("scoped_shots") is None
    sample = _address_site(module, "scoped_draw", str(RANDOM_SAMPLE))
    address = (("call", "scoped_shots#1"), ("call", "scoped_draw#2"))
    expected = tuple((sample, address, (shot,)) for shot in range(2))
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "scoped_shots_run", 1.0) == expected


def test_an_inlined_pure_call_keeps_subsequent_call_ordinals(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("ordinal_pure") is None
    score = _address_site(module, "ordinal_score", str(SCORE_ADD))
    expected = ((score, (("call", "ordinal_score#3"),), ()),)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "ordinal_run", 1.0) == expected


@pytest.mark.parametrize("position", ("bound", "tail"))
def test_retained_call_restores_the_right_surrounding_call_path(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
    position: str,
) -> None:
    module, discharged, round_trip = inlined_address_module
    parent = f"scoped_{position}_parent"
    assert discharged.obstacle(parent) is None
    score = _address_site(module, "ordinal_score", str(SCORE_ADD))
    middle = (("call", "scoped_bound_call#2"),) if position == "bound" else ()
    expected = (
        (score, (("call", f"{parent}#1"), *middle, ("call", "ordinal_score#3")), ()),
    )
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, parent + "_run", 1.0) == expected


def test_retained_tail_call_keeps_its_captured_lexical_history(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("scoped_tap_parent") is None
    trigger = _effect_sites(_computation(module, "scoped_tap_leaf").body, "Tap")[0]
    score = _effect_sites(_effect_clause_body(module, "recording_tap"), "Score")[0]
    source_sites = tuple(
        next(
            term.origin.site_id()
            for term in _terms(_computation(module, name).body)
            if isinstance(term, Call)
        )
        for name in ("scoped_tap_run", "scoped_tap_parent", "scoped_tap_tail")
    )
    key = "/".join((trigger, *(str(site) for site in source_sites)))
    expected = ((score, (("handler-clause", key),), ()),)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "scoped_tap_run", 1.0) == expected


def test_repeated_local_instances_keep_distinct_addresses_at_one_site(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("local_subject") is None
    score = _address_site(module, "local_subject", str(SCORE_ADD))
    after = _address_site(module, "repeated_local_run", str(SCORE_ADD))
    expected = (
        *(
            (score, (("call", f"local_subject#{ordinal}"), ("instance", ordinal)), ())
            for ordinal in (1, 2)
        ),
        (after, (), ()),
    )
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "repeated_local_run", 1.0) == expected


def test_pure_local_instance_allocation_advances_only_its_counter(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("local_pure") is None
    score = _address_site(module, "local_subject", str(SCORE_ADD))
    expected = ((score, (("call", "local_subject#2"), ("instance", 2)), ()),)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "ordinal_local_run", 1.0) == expected


def test_restored_call_and_instance_addresses_keep_their_order_across_capture(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    name = "mixed_local_parent"
    assert discharged.obstacle(name) is None
    parent = _computation(module, name).body
    leaf = _computation(module, "scoped_leaf").body
    (trigger,) = _effect_sites(leaf, "Relay")
    (before, after) = _effect_sites(leaf, "Score")
    (clause,) = _effect_sites(_effect_clause_body(module, "recording_relay"), "Score")
    site = next(
        term.origin.site_id() for term in _terms(parent) if isinstance(term, Call)
    )
    outer = (("call", f"{name}#1"), ("instance", 1))
    active = (*outer, ("call", "scoped_leaf#2"))
    expected = (
        (before, active, ()),
        (clause, (*outer, ("handler-clause", f"{trigger}/{site}")), ()),
        (after, active, ()),
    )
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "mixed_local_run", 1.0) == expected


def test_a_local_instance_blocks_tail_replacement_of_its_caller(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("retained_local_parent") is None
    score = _address_site(module, "ordinal_score", str(SCORE_ADD))
    path = (
        ("call", "retained_local_parent#1"),
        ("instance", 1),
        ("call", "ordinal_score#2"),
    )
    expected = ((score, path, ()),)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "retained_local_run", 1.0) == expected


@pytest.mark.parametrize("outer_omega", (False, True))
def test_a_retained_call_keeps_its_scoped_shots(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
    outer_omega: bool,
) -> None:
    module, discharged, round_trip = inlined_address_module
    assert discharged.obstacle("residual_shot_subject") is None
    sample = _address_site(module, "residual_draw", str(RANDOM_SAMPLE))
    after = _address_site(module, "residual_shot_subject", str(RANDOM_SAMPLE))
    expected = []
    for outer in range(2 if outer_omega else 1):
        parent, first_child = 3 * outer + 1, 3 * outer + 2
        for inner in range(2):
            path = (
                ("call", f"residual_shot_subject#{parent}"),
                ("call", f"residual_draw#{first_child + inner}"),
            )
            shots = (outer, inner) if outer_omega else (inner,)
            expected.append((sample, path, shots))
            expected.append((after, path[:-1], shots))
    name = "residual_nested_run" if outer_omega else "residual_shot_run"
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, name, 1.0) == tuple(expected)


def test_a_retained_call_with_an_interleaved_omega_request_uses_the_reference(
    inlined_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = inlined_address_module
    name = "residual_omega_subject"
    obstacle = discharged.obstacle(name)
    assert obstacle is not None and obstacle.kind == "address"
    assert "`residual_omega_leaf`" in obstacle.reason
    assert _computation(module, name) is _computation(discharged.module, name)
    sample = _address_site(module, "residual_omega_leaf", str(RANDOM_SAMPLE))
    path = (("call", f"{name}#1"), ("call", "residual_omega_leaf#2"))
    expected = tuple((sample, path, (0, shot)) for shot in range(2))
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "residual_omega_call_run", -1.0) == expected


_RETURN_ADDRESS_SOURCE = """
instance second_relay : Relay

handler return_score for Relay : Real -> Real [coverage=total, forwards=none, introduces=!{score}, implementation=authored]
    return x =>
        perform score.add(weight(x))
        return x
    pass(v : Real) resumes 1 =>
        resume(v)

handler return_twice for Fork : Real -> Real [coverage=total, forwards=none, introduces=!{score}, implementation=authored]
    return x =>
        perform score.add(weight(x))
        return x
    split(v : Real) resumes omega =>
        let first <- resume(v)
        let second <- resume(v)
        return first + second

handler return_tap for Relay : Real -> Real [coverage=total, forwards=none, introduces=!{tap}, implementation=authored]
    return x =>
        let y <- perform tap.echo(x)
        return y
    pass(v : Real) resumes 1 =>
        resume(v)

handler tap_forks for Tap : Real -> Real [coverage=total, forwards=none, introduces=!{outside_branch, score}, implementation=authored]
    return x =>
        return x
    echo(v : Real) resumes 1 =>
        let y <- perform outside_branch.split(v)
        perform score.add(weight(y))
        resume(y)

define return_subject(x : Real) : Real !{score} =
    handle second_relay with return_score in
        handle relay with return_score in
            let y <- perform second_relay.pass(x)
            let z <- perform relay.pass(y)
            perform score.add(weight(z))
            return z

define return_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let y <- return_subject(x)
        perform score.add(weight(y))
        return y

define return_shots_subject(x : Real) : Real !{score} =
    handle branch with return_twice in
        let y <- perform branch.split(x)
        return y

define return_shots_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let y <- return_shots_subject(x)
        perform score.add(weight(y))
        return y

define return_capture_subject(x : Real) : Real !{tap} =
    with instance local : Relay in
        handle second_relay with return_tap in
            handle relay with return_tap in
                let y <- perform second_relay.pass(x)
                let z <- perform relay.pass(y)
                return z

define direct_capture_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle tap with recording_tap in
            return_capture_subject(x)

define capture_within(x : Real) : Real !{score} =
    handle tap with recording_tap in
        return_capture_subject(x)

define capture_within_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        capture_within(x)

define one_return_tap(x : Real) : Real !{tap} =
    handle relay with return_tap in
        let y <- perform relay.pass(x)
        return y

define deferred_omega_subject(x : Real) : Real !{outside_branch, score} =
    handle tap with tap_forks in
        one_return_tap(x)

define deferred_omega_run(x : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle outside_branch with guarded_shots in
            deferred_omega_subject(x)
"""


@pytest.fixture(scope="module")
def return_address_module() -> tuple[QiecModule, DischargedModule, QiecModule]:
    module = _module(_ADDRESS_SOURCE + _CLAUSE_SOURCE + _RETURN_ADDRESS_SOURCE)
    discharged = discharge_module(module)
    round_trip = loads(dumps(discharged.module))
    assert isinstance(round_trip, QiecModule)
    return module, discharged, round_trip


@pytest.mark.parametrize("repeated_shots", (False, True))
def test_return_clauses_use_the_installation_address_until_their_answer(
    return_address_module: tuple[QiecModule, DischargedModule, QiecModule],
    repeated_shots: bool,
) -> None:
    module, discharged, round_trip = return_address_module
    suffix = "_shots" if repeated_shots else ""
    subject = "return" + suffix + "_subject"
    root = "return" + suffix + "_run"
    handler_name = "return_twice" if repeated_shots else "return_score"
    handler = next(item for item in module.handlers if item.name == handler_name)
    assert handler.return_clause is not None and discharged.obstacle(subject) is None
    (score,) = _effect_sites(handler.return_clause.body, "Score")
    after = _address_site(module, root, str(SCORE_ADD))
    call = (("call", f"{subject}#1"),)
    visits = ((1, (0,)), (1, (1,))) if repeated_shots else ((2, ()), (1, ()))
    before = (
        ()
        if repeated_shots
        else ((_address_site(module, subject, str(SCORE_ADD)), call, ()),)
    )
    expected = (
        *before,
        *(
            (score, (*call, ("handler-return", f"{handler.id}#{ordinal}")), shots)
            for ordinal, shots in visits
        ),
        (after, (), ()),
    )
    assert len(set(expected)) == len(expected)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, root, 1.0) == expected


@pytest.mark.parametrize("outer_static", (False, True))
def test_a_captured_return_activation_identifies_the_outer_operation_clause(
    return_address_module: tuple[QiecModule, DischargedModule, QiecModule],
    outer_static: bool,
) -> None:
    module, discharged, round_trip = return_address_module
    assert discharged.obstacle("return_capture_subject") is None
    assert discharged.obstacle("capture_within") is None
    assert not any(
        isinstance(term, Handle)
        for term in _terms(_computation(discharged.module, "capture_within").body)
    )
    handler = next(item for item in module.handlers if item.name == "return_tap")
    assert handler.return_clause is not None
    (trigger,) = _effect_sites(handler.return_clause.body, "Tap")
    (score,) = _effect_sites(_effect_clause_body(module, "recording_tap"), "Score")
    caller = "capture_within" if outer_static else "direct_capture_run"
    call = next(
        term.origin.site_id()
        for term in _terms(_computation(module, caller).body)
        if isinstance(term, Call)
    )
    parent = (("call", "capture_within#1"),) if outer_static else ()
    expected = tuple(
        (
            score,
            (
                *parent,
                (
                    "handler-clause",
                    f"{trigger}/{call}/instance:1/handler-return:{handler.id}#{ordinal}",
                ),
            ),
            (),
        )
        for ordinal in (2, 1)
    )
    root = "capture_within_run" if outer_static else caller
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, root, 1.0) == expected


def test_an_interleaved_omega_keeps_the_reference_for_a_deferred_clause_scope(
    return_address_module: tuple[QiecModule, DischargedModule, QiecModule],
) -> None:
    module, discharged, round_trip = return_address_module
    name = "deferred_omega_subject"
    obstacle = discharged.obstacle(name)
    assert obstacle is not None and obstacle.kind == "address"
    assert "Fork.split" in obstacle.reason
    assert _computation(discharged.module, name) is _computation(module, name)
    handler = next(item for item in module.handlers if item.name == "return_tap")
    assert handler.return_clause is not None
    (trigger,) = _effect_sites(handler.return_clause.body, "Tap")
    (score,) = _effect_sites(_effect_clause_body(module, "tap_forks"), "Score")
    call = next(
        term.origin.site_id()
        for term in _terms(_computation(module, name).body)
        if isinstance(term, Call)
    )
    key = f"{trigger}/{call}/handler-return:{handler.id}#1"
    path = (("call", f"{name}#1"), ("handler-clause", key))
    expected = ((score, path, (0,)),)
    for checked in (module, discharged.module, round_trip):
        assert _request_addresses(checked, "deferred_omega_run", -1.0) == expected
        with pytest.raises(
            ExecutionFailure, match="a '1' resumption was invoked more than once"
        ):
            _request_addresses(checked, "deferred_omega_run", 1.0)


# ---------------------------------------------------------------------------
# One computation per way of resuming
# ---------------------------------------------------------------------------

FORMS = (
    HANDLERS
    + """
define linear(a : Real) : Real !{} =
    handle rd with reader(2.0) in
        let w <- perform rd.ask()
        return w + a

define threaded(a : Real) : Real !{} =
    handle st with cell(a) in
        let before <- perform st.get()
        perform st.put(before * 3.0)
        let after <- perform st.get()
        return after - 1.0

define summed(a : Real, b : Real) : Real !{} =
    handle wr with writer(0.5) in
        perform wr.tell(a)
        perform wr.tell(a * b)
        return b

define aborted(a : Real) : Real !{} =
    handle ab with stopped in
        let x <- perform ab.stop(a)
        return x + 100.0

define guard(x : Real) : Real !{ab} =
    if x > 0.0 then
        return x
    else
        let y <- perform ab.stop(x - 1.0)
        return y

define guarded(a : Real, b : Real) : Real !{} =
    handle ab with stopped in
        let x <- guard(a)
        let y <- guard(b)
        return x * y + 10.0

define affine(a : Real) : Real !{} =
    handle mb with checked in
        let x <- perform mb.check(a)
        return x * 2.0

define shots(a : Real, b : Real) : Real !{} =
    handle ch with mixed in
        let x <- perform ch.flip(a)
        let y <- perform ch.flip(b)
        return x * y + x

define per_shot(a : Real) : Real !{} =
    handle ct with counting(1.0, a) in
        let x <- perform ct.flip(a)
        let y <- perform ct.flip(x)
        return x + y

define outside_tail(a : Real) : Real !{} =
    handle af with doubled in
        let x <- perform af.after(a)
        let y <- perform af.after(x + 1.0)
        return x * y

define state_per_shot(a : Real) : Real !{} =
    handle ch with mixed in
        handle st with cell(a) in
            let x <- perform ch.flip(0.75)
            let s <- perform st.get()
            perform st.put(s + x)
            let t <- perform st.get()
            return t

define state_across_shots(a : Real) : Real !{} =
    handle st with cell(a) in
        handle ch with mixed in
            let x <- perform ch.flip(0.75)
            let s <- perform st.get()
            perform st.put(s + x)
            let t <- perform st.get()
            return t

define conditional_put(x : Real) : Unit !{st} =
    if x > 0.0 then
        let s <- perform st.get()
        perform st.put(s + x)
        return unit
    else
        return unit

define conditional(a : Real, b : Real) : Real !{} =
    handle st with cell(0.5) in
        conditional_put(a)
        conditional_put(b)
        conditional_put(a - b)
        let r <- perform st.get()
        return r
"""
)

_VALUES = {
    "linear": lambda a, b: 2.0 + a,
    "threaded": lambda a, b: 3.0 * a - 1.0 + 1.5 * a,
    "summed": lambda a, b: b + 0.5 + a + a * b,
    "aborted": lambda a, b: 0.25 * a,
    "guarded": lambda a, b: (
        a * b + 10.0 if a > 0 and b > 0 else 0.25 * ((b if a > 0 else a) - 1.0)
    ),
    "affine": lambda a, b: 2.0 * a - 0.125 if a > 0 else 1.0 - a,
    "shots": lambda a, b: (
        0.25 * (0.25 * (a * b + a) + 0.75 * (-a * b + a))
        + 0.75 * (0.25 * (-a * b - a) + 0.75 * (a * b - a))
    ),
    "outside_tail": lambda a, b: 0.5 * a * (a + 1.0) + 0.5 * (a + 1.0) + 0.5 * a,
    "state_per_shot": lambda a, b: 1.5 * (a + 0.25 * 0.75 - 0.75 * 0.75),
    "conditional": lambda a, b: (
        1.5 * (0.5 + max(a, 0.0) + max(b, 0.0) + max(a - b, 0.0))
    ),
}


@pytest.mark.parametrize("name", sorted(_VALUES))
@pytest.mark.parametrize("point", _POINTS)
def test_a_form_has_its_value_before_and_after(
    name: str, point: tuple[float, float]
) -> None:
    module = _module(FORMS)
    count = len(_computation(module, name).parameters)
    value = _agree(module, name, point[:count])
    assert value == pytest.approx(_VALUES[name](*point), rel=1e-9)


@pytest.mark.parametrize("name", ["per_shot", "state_across_shots"])
def test_shots_never_share_parameters(name: str) -> None:
    """The second shot of an inner handler reads what the clause names for it,
    while a state handler outside a choice sees the first shot's update."""
    module = _module(FORMS)
    a = 0.3137
    value = _agree(module, name, (a,))
    if name == "state_across_shots":
        first = a + 0.75
        second = first - 0.75
        assert value == pytest.approx(
            0.25 * first + 0.75 * second + 0.5 * second, rel=1e-9
        )


def test_a_guarded_request_becomes_a_conditional() -> None:
    body = _discharged(_module(FORMS), "guarded").body
    conditionals = [term for term in _terms(body) if isinstance(term, If)]
    assert len(conditionals) == 2


def test_conditional_updates_meet_so_the_code_grows_linearly() -> None:
    """Each conditional request is stated once, not once per path."""
    body = _discharged(_module(FORMS), "conditional").body
    assert sum(isinstance(term, If) for term in _terms(body)) == 3


@pytest.mark.parametrize("seed", range(24))
def test_random_programs_have_their_value_before_and_after(seed: int) -> None:
    module = _module(HANDLERS + random_body(seed))
    result = discharge_module(module)
    obstacle = result.obstacle("handled")
    if obstacle is not None:
        # Only the size bound stops an authored, total handler over
        # computations without recursion.
        assert obstacle.kind == "growth"
        return
    for point in _POINTS:
        _agree(module, "handled", point)


def test_the_random_programs_are_mostly_discharged() -> None:
    kinds = [
        discharge_module(_module(HANDLERS + random_body(seed))).obstacle("handled")
        for seed in range(24)
    ]
    assert sum(item is None for item in kinds) >= 18


# ---------------------------------------------------------------------------
# What stays, and why
# ---------------------------------------------------------------------------


def test_requests_outside_the_handler_and_calls_that_make_none_stay() -> None:
    module = _module(
        HANDLERS
        + """
define twice(x : Real) : Real !{} =
    return x + x

define mixed_row(a : Real) : Real !{rd} =
    handle st with cell(a) in
        let r <- perform rd.ask()
        let t <- twice(r)
        perform st.put(t)
        let s <- perform st.get()
        return s + a
"""
    )
    body = _discharged(module, "mixed_row").body
    assert [type(term).__name__ for term in _terms(body)].count("Perform") == 1
    assert [term.name for term in _terms(body) if isinstance(term, Call)] == ["twice"]


def test_a_request_of_a_clause_body_stays_and_names_its_clause() -> None:
    module = _module(
        HANDLERS
        + """
handler logging for Maybe : Real -> Real [coverage=total, forwards=none, introduces=!{wr}, implementation=authored]
    return x =>
        return x
    check(v : Real) resumes 1 =>
        perform wr.tell(v)
        resume(v)

define logged(a : Real) : Real !{wr} =
    handle mb with logging in
        let x <- perform mb.check(a)
        return x
"""
    )
    _discharged(module, "logged")
    result = discharge_module(module)
    (request,) = [
        term.request
        for term in _terms(_computation(result.module, "logged").body)
        if isinstance(term, Perform)
    ]
    assert result.requester(request) == "clause `check` of handler `logging`"


_OBSTACLES = {
    "foreign": (
        """
handler outer for Maybe : Real -> Real [coverage=total, forwards=none, implementation=foreign]
    check(v : Real) resumes 1

define subject(a : Real) : Real !{} =
    handle mb with outer in
        let x <- perform mb.check(a)
        return x
""",
        "installs the foreign handler `outer`, whose clause `check` a runtime "
        "provider implements",
    ),
    "forwarding": (
        """
handler half for State[Real] : Real -> Real [coverage=partial, forwards=unknown, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(1.0)

define subject(a : Real) : Real !{st} =
    handle st with half in
        perform st.put(a)
        let x <- perform st.get()
        return x
""",
        "requests `put` under the handler `half`, which has no clause `put` and "
        "forwards it",
    ),
    "recursion": (
        """
define loop(n : Int, x : Real) : Real !{rd} =
    if n <= 0 then
        return x
    else
        let r <- perform rd.ask()
        loop(n - 1, x + r)

define subject(a : Real) : Real !{} =
    handle rd with reader(a) in
        let x <- loop(3, a)
        return x
""",
        "reaches a recursive call of `loop` that requests from the handler "
        "`reader`, so how often its clauses resume depends on the values the call",
    ),
    "case": (
        """
family Outcome : Type
    constructor Won : Real -> Outcome
    constructor Lost : Outcome

define score(out : Outcome) : Real !{ch} =
    case out motive => Real
        Won(x) =>
            let y <- perform ch.flip(x)
            return y
        Lost =>
            return 0.0

define subject(out : Outcome) : Real !{} =
    handle ch with mixed in
        let k <- perform ch.flip(1.0)
        let x <- score(out)
        return x + k
""",
        "branches with `case` on a value known only at run time",
    ),
    "growth": (
        """
define subject(a : Real) : Real !{} =
    handle ch with mixed in
"""
        + "\n".join(
            f"        let x{k} <- perform ch.flip({'a' if k == 0 else f'x{k - 1}'})"
            for k in range(12)
        )
        + """
        return x11
""",
        f"has more than {DISCHARGE_STATEMENTS} statements once its handlers are "
        "discharged",
    ),
}


@pytest.mark.parametrize("kind", sorted(_OBSTACLES))
def test_an_obstacle_names_the_handler_and_the_clause(kind: str) -> None:
    source, reason = _OBSTACLES[kind]
    module = _module(HANDLERS + source)
    obstacle = discharge_computation(_computation(module, "subject"), module)
    assert isinstance(obstacle, DischargeObstacle)
    assert obstacle.kind == kind
    assert reason in obstacle.reason
    result = discharge_module(module)
    assert result.obstacle("subject") == obstacle
    # The computation is left as it is.
    assert _computation(result.module, "subject") == _computation(module, "subject")


def test_a_forwarded_request_an_outer_handler_answers_is_discharged() -> None:
    module = _module(
        HANDLERS
        + """
handler half for State[Real] : Real -> Real [coverage=partial, forwards=unknown, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(1.0)

define subject(a : Real) : Real !{} =
    handle st with cell(a) in
        handle st with half in
            perform st.put(3.0 * a)
            let x <- perform st.get()
            return x
"""
    )
    assert _agree(module, "subject", (0.25,)) == pytest.approx(1.0 + 0.5 * 0.75)


def test_the_branches_of_a_case_meet_around_the_handler() -> None:
    """A branch a clause ends and one that finishes are one value after it."""
    module = _module(
        HANDLERS
        + """
family Outcome : Type
    constructor Won : Real -> Outcome
    constructor Lost : Outcome

define points(out : Outcome) : Real !{ab} =
    case out motive => Real
        Won(x) =>
            return x
        Lost =>
            let y <- perform ab.stop(2.0)
            return y

define scored(out : Outcome) : Real !{} =
    handle ab with stopped in
        let x <- points(out)
        return x + 1.0

define subject(a : Real) : Real !{} =
    let won <- scored(construct Won(a) as Outcome)
    let lost <- scored(construct Lost() as Outcome)
    return won * 10.0 + lost
"""
    )
    body = _discharged(module, "scored").body
    assert sum(isinstance(term, Case) for term in _terms(body)) == 1
    for checked in (module, discharge_module(module).module):
        assert run_named(checked, "subject", (0.25,)).value == pytest.approx(13.0)


def test_a_request_in_one_branch_of_a_case_that_resumes_twice_is_discharged() -> None:
    """Both shots and the other branch arrive around the handler as one value."""
    module = _module(
        HANDLERS
        + """
family Outcome : Type
    constructor Won : Real -> Outcome
    constructor Lost : Outcome

define score(out : Outcome) : Real !{ch} =
    case out motive => Real
        Won(x) =>
            let y <- perform ch.flip(x)
            return y
        Lost =>
            return 0.5

define flipped(out : Outcome) : Real !{} =
    handle ch with mixed in
        let x <- score(out)
        return x + 1.0

define subject(a : Real) : Real !{} =
    let won <- flipped(construct Won(a) as Outcome)
    let lost <- flipped(construct Lost() as Outcome)
    return won * 10.0 + lost
"""
    )
    _discharged(module, "flipped")
    expected = 10.0 * (0.25 * 1.25 + 0.75 * 0.75) + 1.5
    for checked in (module, discharge_module(module).module):
        assert run_named(checked, "subject", (0.25,)).value == pytest.approx(expected)


def test_branches_meet_at_the_handler_whose_clause_ends_one() -> None:
    """Two shots of a choice, each ending early or not, stay two conditionals."""
    module = _module(
        HANDLERS
        + """
define guard(x : Real) : Real !{ab} =
    if x > 0.0 then
        return x
    else
        let y <- perform ab.stop(x)
        return y

define subject(a : Real) : Real !{} =
    handle ch with mixed in
        handle ab with stopped in
            let x <- perform ch.flip(a)
            let y <- guard(x)
            return y * y
"""
    )
    body = _discharged(module, "subject").body
    assert sum(isinstance(term, If) for term in _terms(body)) == 2
    for a in (0.5, -0.5):
        _agree(module, "subject", (a,))


def test_a_case_on_a_constructor_selects_its_branch() -> None:
    module = _module(
        HANDLERS
        + """
family Outcome : Type
    constructor Won : Real -> Outcome
    constructor Lost : Outcome

effect Play
    play : Real -> Outcome

instance game : Play

handler winning for Play : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    play(v : Real) resumes 1 =>
        resume(construct Won(v + 1.0) as Outcome)

define points(out : Outcome) : Real !{ab} =
    case out motive => Real
        Won(x) =>
            return x
        Lost =>
            let y <- perform ab.stop(0.0)
            return y

define subject(a : Real) : Real !{} =
    handle ab with stopped in
        handle game with winning in
            let out <- perform game.play(a)
            let x <- points(out)
            return 2.0 * x
"""
    )
    body = _discharged(module, "subject").body
    assert not any(isinstance(term, Case | If) for term in _terms(body))
    assert _agree(module, "subject", (0.25,)) == pytest.approx(2.5)


def test_polymorphic_handlers_and_operations_are_instantiated() -> None:
    module = _module(
        """
effect Box[S : Type]
    swap : S -> S
    keep[B : Type] : B -> B

handler held[S : Type, A : Type] for Box[S] : A -> A * S with (s : S) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, s)
    swap(v : S) resumes 1 =>
        resume(s) with (v)
    keep[B : Type](b : B) resumes 1 =>
        resume(b)

define pair[T : Type](x : T, y : T) : T * T !{} =
    with instance b : Box[T] in
        handle b with held[T, T](x) in
            let old <- perform b.swap(y)
            let same <- perform b.keep[T](old)
            return same

define subject(a : Real) : Real !{} =
    with instance b : Box[Real] in
        with instance c : Box[Int] in
            let r <- inner(a)
            return r

define inner(a : Real) : Real !{} =
    let p <- pair[Real](a, 2.0 * a)
    return p[0] + 10.0 * p[1]
"""
    )
    registry = validate_module(module)
    result = discharge_module(module)
    assert not result.obstacles
    pair = _computation(result.module, "pair")
    context = CheckContext()
    for parameter in pair.parameters:
        context = context.extend(parameter)
    assert infer_computation(pair.body, registry, context).result == pair.type.result
    assert sum(isinstance(term, InstanceAddress) for term in _terms(pair.body)) == 1
    for checked in (module, result.module):
        assert run_named(checked, "inner", (0.5,)).value == pytest.approx(10.5)


def test_two_instances_of_one_interface_stay_apart() -> None:
    module = _module(
        HANDLERS
        + """
define subject(a : Real, b : Real) : Real !{} =
    with instance left : State[Real] in
        with instance right : State[Real] in
            handle left with cell(a) in
                handle right with cell(b) in
                    let l <- perform left.get()
                    let r <- perform right.get()
                    perform left.put(r)
                    perform right.put(l * 2.0)
                    let x <- perform left.get()
                    let y <- perform right.get()
                    return x - y
"""
    )
    a, b = 0.3137, -0.7171
    value = _agree(module, "subject", (a, b))
    assert value == pytest.approx(b - 2.0 * a + a + 0.5 * b)


def test_the_bound_holds_of_the_body_and_not_of_the_steps_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the branches adds statements after reduction has finished.

    This case emits thirteen statements during reduction and has
    fourteen in its closed body.
    """
    module = _module(HANDLERS + random_body(42))
    monkeypatch.setattr(discharge, "DISCHARGE_STATEMENTS", 14)
    rewritten = _discharged(module, "handled")
    assert sum(not isinstance(term, Bind) for term in _terms(rewritten.body)) == 14
    monkeypatch.setattr(discharge, "DISCHARGE_STATEMENTS", 13)
    obstacle = discharge_module(module).obstacle("handled")
    assert obstacle is not None
    assert obstacle.kind == "growth"


def test_a_module_without_handlers_is_returned_as_it_is() -> None:
    module = _module(
        """
define bump(x : Real) : Real !{} =
    return x + 1.0
"""
    )
    assert discharge_module(module).module is module
