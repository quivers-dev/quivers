"""Recursive state specialization against independent values and call orders."""

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    REAL,
    AttachmentId,
    AttachmentRef,
    Bind,
    Call,
    ExecutionFailure,
    Handle,
    LetBinding,
    LetValue,
    If,
    Local,
    Perform,
    ResumptionGrade,
    StateSpecializationLimit,
    StateSpecializationUnsupported,
    QiecModule,
    Return,
    Var,
    called_computations,
    discharge_module,
    dumps,
    loads,
    run_named,
    specialize_recursive_state,
    validate_module,
)

_SOURCE = """
effect Counter
    inspect_value : Unit -> Real
    update_value : Real -> Real
    skip_value : Real -> Real

instance counter : Counter
instance other : Counter

define mark_inspect(left : Real, right : Real) : Real !{} =
    return 10.0 * left + right

define mark_update(amount : Real) : Real !{} =
    return amount

define finish_value(value : Real, left : Real, right : Real) : (Real * Real * Real) !{} =
    return (2.0 * value + left - right, left, right)

handler threaded for Counter : Real -> (Real * Real * Real) with (left : Real, right : Real) [coverage=total, forwards=none, implementation=authored]
    return value =>
        finish_value(value, left, right)
    inspect_value resumes 1 =>
        let value <- mark_inspect(left, right)
        resume(value)
    update_value(amount : Real) resumes 1 =>
        let direction <- mark_update(amount)
        if direction > 0.0 then
            resume(left - right) with (right + direction, left - direction)
        else
            resume(right - left) with (left + direction, right + 2.0 * direction)
    skip_value(ignored : Real) resumes 1 =>
        if true then
            resume(left)
        else
            let answer <- resume(left)
            return answer

define even_walk(n : Real) : Real !{counter} =
    if n > 0.0 then
        let prior <- perform counter.inspect_value()
        let step <- perform counter.update_value(n)
        let nested <- odd_walk(n - 1.0)
        let late <- perform counter.update_value(0.0 - n)
        return 2.0 * nested + 3.0 * prior + step + late
    else
        perform counter.inspect_value()

define odd_walk(n : Real) : Real !{counter} =
    if n > 0.0 then
        let prior <- perform counter.inspect_value()
        let step <- perform counter.update_value(n)
        let nested <- even_walk(n - 1.0)
        let late <- perform counter.update_value(0.0 - n)
        return 2.0 * nested + 3.0 * prior + step + late
    else
        perform counter.inspect_value()

define execute(n : Real, left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        even_walk(n)

define tail_walk(n : Real) : Real !{counter} =
    if n > 0.0 then
        let ignored <- perform counter.update_value(n)
        tail_walk(n - 1.0)
    else
        perform counter.inspect_value()

define tail_execute(n : Real, left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        tail_walk(n)

define return_without_request() : Real !{counter} =
    if false then
        perform counter.inspect_value()
    else
        return 79.0

define return_only(left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        return_without_request()

define ordinary_step(left : Real, right : Real) : Real !{counter} =
    if false then
        perform counter.inspect_value()
    else
        mark_inspect(right, left)

define ordinary(left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        ordinary_step(left, right)

define direct(amount : Real, left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        perform counter.skip_value(amount)

define nested_inner(left : Real, right : Real) : (Real * Real * Real) !{} =
    handle other with threaded(left, right) in
        perform other.inspect_value()

define nested_scope(left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        let nested <- nested_inner(left, right)
        perform counter.inspect_value()

define loops(n : Real) : Real !{counter} =
    let ignored <- perform counter.update_value(n)
    loops(n)

define diverges(left : Real, right : Real) : (Real * Real * Real) !{} =
    handle counter with threaded(left, right) in
        loops(1.0)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    lowered = Compiler(parse(_SOURCE), module_name="recursive-state-test").qiec_module
    assert lowered is not None
    return lowered


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _specialize(module, name, **options):
    return specialize_recursive_state(module, _entry(module, name).id, **options)


def _replace_body(module, owner, body):
    return module.with_(
        computations=tuple(
            item.with_(body=body) if item.id == owner.id else item
            for item in module.computations
        )
    )


def _triple(value: object) -> tuple[float, ...]:
    assert isinstance(value, tuple)
    return tuple(map(float, value))


def _oracle(n, left, right):
    """Simulate source chronology with an explicit mathematical unwind table."""
    frames = []
    events = []
    for depth, amount in enumerate(range(n, 0, -1)):
        events.extend(
            [
                "odd_walk" if depth % 2 else "even_walk",
                "inspect_value",
                "mark_inspect",
                "update_value",
                "mark_update",
            ]
        )
        frames.append((amount, 10 * left + right, left - right))
        left, right = right + amount, left - amount
    events.extend(
        [
            "odd_walk" if n % 2 else "even_walk",
            "inspect_value",
            "mark_inspect",
        ]
    )
    value = 10 * left + right
    while frames:
        amount, prior, updated = frames.pop()
        events.extend(["update_value", "mark_update"])
        value = 2 * value + 3 * prior + updated + (right - left)
        left, right = left - amount, right - 2 * amount
    events.append("finish_value")
    return (2 * value + left - right, left, right), events


def _orders(module, execution, aliases):
    operations = {
        str(operation.id): operation.name
        for effect in module.effects
        for operation in effect.operations
    }
    source, target = [], []
    for event in execution.trace:
        if event.event == "operation.requested":
            source.append(operations[event.detail["operation"]])
        if event.event == "call.entered":
            name = event.detail["computation"]
            if name in aliases:
                target.append(aliases[name])
            elif name in {
                "even_walk",
                "odd_walk",
                "mark_inspect",
                "mark_update",
                "finish_value",
            }:
                source.append(name)
                target.append(name)
    return source, target


def _aliases(module, output):
    by_id = {str(item.id): item.name for item in module.computations}
    by_id.update(
        (str(operation.id), operation.name)
        for effect in module.effects
        for operation in effect.operations
    )
    known = {item.id for item in module.computations}
    return {
        item.name: by_id[key]
        for item in output.computations
        if item.id not in known
        for role, key in [item.origin.structural_path[-2:]]
        if role in {"computation", "operation"}
    }


@pytest.mark.parametrize("n", [0, 1, 2, 5, 9])
@pytest.mark.parametrize(("left", "right"), [(-3, 2), (0, 0), (7, -4)])
def test_mutual_recursion_preserves_updates_and_order(module, n, left, right):
    output = _specialize(module, "execute")
    expected, events = _oracle(n, left, right)
    source = run_named(module, "execute", (n, left, right))
    target = run_named(output, "execute", (n, left, right))
    assert _triple(source.value) == expected
    assert _triple(target.value) == expected
    assert _orders(module, source, {})[0] == events
    assert _orders(module, target, _aliases(module, output))[1] == events
    assert all(event.event != "operation.requested" for event in target.trace)
    assert all(event.event != "handler.entered" for event in target.trace)


def _terms(body):
    yield body
    if isinstance(body, Bind):
        yield from _terms(body.first)
        yield from _terms(body.then)
    if isinstance(body, If):
        yield from _terms(body.then)
        yield from _terms(body.otherwise)


def test_return_action_occurs_only_at_installation_boundary(module):
    output = _specialize(module, "execute")
    original_ids = {item.id for item in module.computations}
    added = [item for item in output.computations if item.id not in original_ids]
    roles = {item.id: item.origin.structural_path[-2] for item in added}
    recursive = [item for item in added if roles[item.id] == "computation"]
    final = next(item for item in added if roles[item.id] == "return")
    assert len(recursive) == 2
    assert all(
        called_computations(item.body) & {other.id}
        for item, other in (recursive, recursive[::-1])
    )
    assert all(final.id not in called_computations(item.body) for item in recursive)
    final_references = [
        term
        for term in _terms(_entry(output, "execute").body)
        if isinstance(term, Call) and term.callee == final.id
    ]
    assert len(final_references) == 1


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("return_only", (2, 3), (157, 2, 3)),
        ("ordinary", (2, 3), (63, 2, 3)),
        ("direct", (1000, 2, 3), (3, 2, 3)),
        ("tail_execute", (0, 2, 3), (45, 2, 3)),
        ("tail_execute", (1, 2, 3), (85, 4, 1)),
    ],
)
def test_exit_cases_without_non_tail_recursive_work(module, name, args, expected):
    output = _specialize(module, name)
    assert _triple(run_named(module, name, args).value) == expected
    assert _triple(run_named(output, name, args).value) == expected


def test_failure_in_an_unused_request_argument_remains_strict(module):
    owner = _entry(module, "direct")
    installation = owner.body
    assert isinstance(installation, Handle)
    perform = installation.computation
    assert isinstance(perform, Perform)
    missing = AttachmentRef(
        attachment=AttachmentId.derive("missing-state-arg"), type=REAL
    )
    body = installation.with_(
        computation=perform.with_(request=perform.request.with_(arguments=(missing,)))
    )
    original = _replace_body(module, owner, body)
    output = _specialize(original, "direct")
    for candidate in (original, output):
        with pytest.raises(ExecutionFailure, match="attachment") as error:
            run_named(candidate, "direct", (1000, 2, 3))
        assert error.value.diagnostic.code == "qiec-run-evaluation"


def test_strict_initial_state_arguments_run_in_declaration_order(module):
    owner = _entry(module, "direct")
    installation = owner.body
    assert isinstance(installation, Handle)
    first = AttachmentRef(
        attachment=AttachmentId.derive("missing-first-initial-argument"), type=REAL
    )
    second = AttachmentRef(
        attachment=AttachmentId.derive("missing-second-initial-argument"), type=REAL
    )
    original = _replace_body(
        module, owner, installation.with_(arguments=(first, second))
    )
    output = _specialize(original, "direct")
    for candidate in (original, output):
        with pytest.raises(ExecutionFailure, match=str(first.attachment)) as error:
            run_named(candidate, "direct", (1000, 2, 3))
        assert error.value.diagnostic.code == "qiec-run-evaluation"


def test_recursive_code_size_exact_construction_limit_and_wire_roundtrip(module):
    entry = _entry(module, "tail_execute")
    before = dumps(module)
    output = _specialize(module, "tail_execute")
    known = {item.id for item in module.computations}
    new = [item for item in output.computations if item.id not in known]
    generated = [_entry(output, "tail_execute"), *new]
    assert len(new) == 5
    assert sum(item.origin.structural_path[-2] == "computation" for item in new) == 1
    total = sum(sum(1 for _ in _terms(item.body)) for item in generated)
    with pytest.raises(StateSpecializationLimit, match=f"bound of {total - 1}"):
        specialize_recursive_state(module, entry.id, max_terms=total - 1)
    assert dumps(
        specialize_recursive_state(module, entry.id, max_terms=total)
    ) == dumps(output)
    assert dumps(module) == before
    obstacle = discharge_module(module).obstacle("tail_execute")
    assert obstacle is not None and obstacle.kind == "recursion"
    round_tripped = loads(dumps(output))
    assert isinstance(round_tripped, QiecModule)
    validate_module(round_tripped)
    assert run_named(
        round_tripped, "tail_execute", (200, 2, 3), record_trace=False
    ).value == (
        run_named(module, "tail_execute", (200, 2, 3), record_trace=False).value
    )


def test_divergence_obeys_runtime_fuel(module):
    output = _specialize(module, "diverges")
    for candidate in (module, output):
        with pytest.raises(ExecutionFailure) as error:
            run_named(candidate, "diverges", (2, 3), fuel=256, record_trace=False)
        assert error.value.diagnostic.code == "qiec-run-fuel"


def test_nested_installation_and_unrestricted_grade_are_refused(module):
    with pytest.raises(StateSpecializationUnsupported, match="Handle"):
        _specialize(module, "nested_scope")
    handler = module.handlers[0]
    changed = handler.with_(
        clauses=(
            handler.clauses[0].with_(grade=ResumptionGrade.UNRESTRICTED),
            *handler.clauses[1:],
        )
    )
    candidate = module.with_(handlers=(changed,))
    validate_module(candidate)
    with pytest.raises(StateSpecializationUnsupported, match="linear grade"):
        _specialize(candidate, "execute")


def test_inserted_locals_do_not_capture_existing_binders(module):
    owner = _entry(module, "return_only")
    installation = owner.body
    assert isinstance(installation, Handle)
    binder = Local(name="__state_1", type=REAL)
    value_binder = Local(name="__state_2", type=REAL)
    value = LetValue(
        bindings=(LetBinding(binder=value_binder, value=Var(local=binder)),),
        body=Var(local=value_binder),
        result_type=REAL,
    )
    body = installation.with_(
        computation=Bind(
            binder=binder,
            first=installation.computation,
            then=Return(value=value),
        )
    )
    original = _replace_body(module, owner, body)
    output = _specialize(original, "return_only")
    assert _triple(run_named(output, "return_only", (2, 3)).value) == (
        157,
        2,
        3,
    )
