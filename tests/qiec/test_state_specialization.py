"""Recursive state specialization against independent values and call orders."""

from collections import Counter
from fractions import Fraction

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EMPTY_ROW,
    REAL,
    SCORE_ADD,
    AttachmentId,
    AttachmentRef,
    Bind,
    Call,
    Case,
    DischargeObstacle,
    ExecutionFailure,
    Handle,
    HandlerReturnAddress,
    JvpBoundaryError,
    JvpUnsupported,
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
    RuntimeConfiguration,
    RuntimeSelection,
    Var,
    WithTraceAddresses,
    called_computations,
    discharge_computation,
    discharge_module,
    dumps,
    exact_gradient,
    exact_jvp,
    loads,
    run_named,
    specialize_recursive_state,
    validate_module,
)
from quivers.qiec.differentiation import _ScalarDerivativeEvaluator
from quivers.qiec.terms import _RETURN_ADDRESS
from quivers.transpile import transpile
from quivers.transpile._entry import _emit_target
from quivers.transpile._pipeline import parser_registry
from quivers.transpile.ir import IRProgram
from quivers.transpile.qiec_ir import capabilities_for_target, project_for_target
from quivers.transpile.renderers import PyroRenderer

_SOURCE = """
effect Counter
    inspect_value : Unit -> Real
    update_value : Real -> Real
    skip_value : Real -> Real

instance counter : Counter
instance other : Counter
instance score : Score

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

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define outside_score(value : Real) : Real !{score} =
    perform score.add(weight(value))
    return value

define scored(n : Real, left : Real, right : Real) : (Real * LogWeight) !{} =
    handle score with accumulate in
        let result <- execute(n, left, right)
        outside_score(left)
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
    known = {item.id for item in module.computations}
    return {
        item.name: by_id[key]
        for item in output.computations
        if item.id not in known
        for role, key in [item.origin.structural_path[-2:]]
        if role == "computation"
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
    source_calls = [
        item for item in events if item not in {"inspect_value", "update_value"}
    ]
    assert _orders(module, target, _aliases(module, output))[1] == source_calls
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
    if isinstance(body, Case):
        for branch in body.branches:
            yield from _terms(branch.body)


def test_return_action_occurs_only_at_installation_boundary(module):
    output = _specialize(module, "execute")
    original_ids = {item.id for item in module.computations}
    recursive = [item for item in output.computations if item.id not in original_ids]
    final = _entry(module, "finish_value")
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
    assert len(new) == 1
    assert sum(item.origin.structural_path[-2] == "computation" for item in new) == 1
    total = sum(sum(1 for _ in _terms(item.body)) for item in generated)
    with pytest.raises(StateSpecializationLimit, match=f"bound of {total - 1}"):
        specialize_recursive_state(module, entry.id, max_terms=total - 1)
    assert dumps(
        specialize_recursive_state(module, entry.id, max_terms=total)
    ) == dumps(output)
    assert dumps(module) == before
    plain = discharge_computation(entry, module)
    assert isinstance(plain, DischargeObstacle) and plain.kind == "recursion"
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
    integrated = discharge_module(module)
    assert integrated.obstacle("diverges") is None
    for candidate in (module, output, integrated.module):
        with pytest.raises(ExecutionFailure) as error:
            run_named(candidate, "diverges", (2, 3), fuel=256, record_trace=False)
        assert error.value.diagnostic.code == "qiec-run-fuel"


def test_discharge_report_names_actual_state_companions(module):
    original = {item.id for item in module.computations}
    discharged = discharge_module(module)
    rewrite = next(
        item
        for item in discharged.rewrites
        if item.computation == _entry(module, "execute").id
    )
    assert rewrite.method == "recursive-state"
    actual = _entry(discharged.module, "execute")
    declarations = {item.id: item for item in discharged.module.computations}
    pending = [actual.id]
    reachable = set()
    while pending:
        identity = pending.pop()
        if identity not in reachable:
            reachable.add(identity)
            pending.extend(called_computations(declarations[identity].body))
    new = reachable - original
    assert len(new) == 2
    assert set(rewrite.companions) == new
    assert all(declarations[identity].type.effects == EMPTY_ROW for identity in new)


@pytest.mark.parametrize("n", [0, 1, 4])
def test_open_caller_keeps_the_address_of_a_later_score(module, n):
    output = _specialize(module, "execute")
    integrated = discharge_module(module)
    assert integrated.obstacle("execute") is None
    obstacle = integrated.obstacle("scored")
    assert obstacle is not None and obstacle.kind == "foreign"
    scorer = _entry(module, "outside_score")
    site = next(
        term.request.origin.static_site
        for term in _terms(scorer.body)
        if isinstance(term, Perform)
    )
    # execute, n+1 walking calls, n+1 read prefixes, 2n update prefixes,
    # the one return-action call, then outside_score.
    expected = (str(site), (("call", f"outside_score#{4 * n + 5}"),), ())
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core",
                options={"handlers": {"accumulate": {"kind": "score"}}},
            ),
        )
    )
    for candidate in (module, output, integrated.module):
        trace = run_named(candidate, "scored", (n, 2, 3), runtime=runtime).trace
        addresses = [
            event.detail["address"]
            for event in trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(SCORE_ADD)
        ]
        assert addresses == [expected]


def test_earlier_installation_counts_before_a_retained_return_address(module):
    scorer = _entry(module, "outside_score")
    handler = next(item for item in module.handlers if item.name == "threaded")
    local = Local(name="later_return", type=_RETURN_ADDRESS)
    body = Bind(
        binder=local,
        first=HandlerReturnAddress(handler=handler.id),
        then=WithTraceAddresses(addresses=(Var(local=local),), body=scorer.body),
    )
    original = _replace_body(module, scorer, body)
    output = discharge_module(original).module
    site = next(
        term.request.origin.static_site
        for term in _terms(scorer.body)
        if isinstance(term, Perform)
    )
    expected = (
        str(site),
        (("call", "outside_score#5"), ("handler-return", f"{handler.id}#2")),
        (),
    )
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core",
                options={"handlers": {"accumulate": {"kind": "score"}}},
            ),
        )
    )
    for candidate in (original, output):
        trace = run_named(candidate, "scored", (0, 2, 3), runtime=runtime).trace
        addresses = [
            event.detail["address"]
            for event in trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(SCORE_ADD)
        ]
        assert addresses == [expected]


def test_companions_execute_through_the_generated_python_runtime(module):
    discharged = discharge_module(module)
    assert discharged.obstacle("execute") is None
    capabilities = capabilities_for_target("pyro")
    projection = project_for_target(
        module,
        discharged,
        "pyro",
        capabilities=capabilities.with_(features=capabilities.features - {"handle"}),
    )
    ir = IRProgram(name="state", inputs=(), body=(), module=projection)
    rendered = PyroRenderer().render(ir)
    namespace = {}
    exec(bytes(parser_registry().emit_pretty("python", rendered)), namespace)
    for n in (0, 1, 7):
        expected, _ = _oracle(n, 2, 3)
        assert _triple(namespace["qiec_execute"](n, 2, 3)) == expected


_CONSTRUCTOR_STATE = """
family Path : Type
    constructor End : Path
    constructor Step : Real * Real * Path -> Path

effect Cell
    read : Unit -> Real
    write : Real -> Unit

instance cell : Cell
instance score : Score

handler state for Cell : Real -> Real with (current : Real) [coverage=total, forwards=none, implementation=authored]
    return value =>
        return 2.0 * value + current
    read resumes 1 =>
        resume(current)
    write(value : Real) resumes 1 =>
        resume() with (value)

define schedule(p : Real, q : Real) : Path !{} =
    if p > 0.0 then
        let end = construct End() as Path
        let rest = construct Step(2.0 * p, q + 1.0, end) as Path
        return construct Step(p, q, rest) as Path
    else
        return construct End() as Path

define prepend(amount : Real, scale : Real, rest : Path) : Path !{} =
    return construct Step(amount, scale, rest) as Path

define walk(path : Path) : Real !{cell} =
    case path motive => Real
        End =>
            perform cell.read()
        Step(amount, scale, rest) =>
            let prior <- perform cell.read()
            perform cell.write(scale * prior + amount)
            let inner <- walk(rest)
            let late <- perform cell.read()
            return inner + prior + late

define solve(initial : Real, p : Real, q : Real) : Real !{} =
    handle cell with state(initial) in
        let path <- schedule(p, q)
        walk(path)

define from_path(path : Path, initial : Real) : Real !{} =
    handle cell with state(initial) in
        walk(path)

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define mark(value : Real) : Real !{score} =
    perform score.add(weight(value))
    return value

define scored(initial : Real, p : Real, q : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        let value <- solve(initial, p, q)
        mark(value)
"""
_CASE_INPUTS = (
    (Fraction(1), Fraction(-2), Fraction(2)),
    (Fraction(1), Fraction(2), Fraction(2)),
    (Fraction(-3, 2), Fraction(1, 2), Fraction(3, 4)),
)


def _constructor_polynomial(initial, p, q):
    if p < 0:
        return 0, 3 * initial, (Fraction(3), Fraction(), Fraction())
    assert p > 0
    # The two Step values have distinct amount and scale fields. The
    # terminal value and both post-call reads use the final state.
    value = 7 * ((q + 1) * (q * initial + p) + 2 * p) + 2 * (initial + q * initial + p)
    gradient = (
        7 * q * (q + 1) + 2 + 2 * q,
        7 * q + 23,
        7 * ((2 * q + 1) * initial + p) + 2 * initial,
    )
    return 2, value, gradient


@pytest.fixture(scope="module")
def constructor_state():
    compiler = Compiler(parse(_CONSTRUCTOR_STATE), module_name="recursive-case-state")
    source = compiler.qiec_module
    assert source is not None
    direct = _specialize(source, "solve")
    selected = discharge_module(source)
    assert selected.obstacle("solve") is None
    rewrite = next(
        item
        for item in selected.rewrites
        if item.computation == _entry(source, "solve").id
    )
    assert rewrite.method == "recursive-state" and len(rewrite.companions) == 1
    staged = selected.module
    restored = loads(dumps(staged))
    assert isinstance(restored, QiecModule) and restored == staged
    return compiler, source, direct, staged, restored


def _constructor_trace(source, observed, point, runtime):
    mark = _entry(source, "mark").body
    assert isinstance(mark, Bind) and isinstance(mark.first, Perform)
    score_site = str(mark.first.request.origin.static_site)
    run = run_named(observed, "scored", tuple(map(float, point)), runtime=runtime)
    aliases = _aliases(source, observed)
    calls = []
    scores = []
    operations = []
    cell = next(effect for effect in source.effects if effect.ref.name == "Cell")
    names = {str(operation.id): operation.name for operation in cell.operations}
    for event in run.trace:
        if event.event == "call.entered":
            name = str(event.detail["computation"])
            calls.append(aliases.get(name, name))
        if event.event == "operation.requested":
            operation = str(event.detail["operation"])
            if operation == str(SCORE_ADD):
                scores.append(event.detail["address"])
            elif operation in names:
                operations.append(names[operation])
    return _triple(run.value), calls, scores, operations, score_site


def _scoring_runtime():
    return RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core", options={"handlers": {"accumulate": {"kind": "score"}}}
            ),
        )
    )


def test_constructor_state_preserves_selected_count_value_and_derivatives(
    constructor_state, monkeypatch
):
    _, source, direct, staged, restored = constructor_state
    primitive = _ScalarDerivativeEvaluator.primitive
    operations = Counter()

    def counted(self, term, environment):
        operations[term.name] += 1
        return primitive(self, term, environment)

    monkeypatch.setattr(_ScalarDerivativeEvaluator, "primitive", counted)
    for point in _CASE_INPUTS:
        count, value, gradient = _constructor_polynomial(*point)
        expected_arithmetic = (
            {"gt_real": 1, "mul_real": 4, "add_real": 8}
            if count
            else {"gt_real": 1, "mul_real": 1, "add_real": 1}
        )
        for graph in (source, direct, staged, restored):
            operations.clear()
            result = exact_gradient(graph, "solve", point)
            assert (result.value, result.gradient) == (value, gradient)
            assert operations == expected_arithmetic
            for index, basis in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                jvp = exact_jvp(graph, "solve", point, basis)
                assert (jvp.value, jvp.tangent) == (value, gradient[index])
            observed, calls, scores, active, site = _constructor_trace(
                source, graph, point, _scoring_runtime()
            )
            assert observed == (value, value)
            assert calls == ["solve", "schedule", *(["walk"] * (count + 1)), "mark"]
            assert scores == [(site, (("call", f"mark#{count + 4}"),), ())]
            assert active == (
                ["read", "write"] * count + ["read"] * (count + 1)
                if graph is source
                else []
            )


def test_constructor_state_boundary_is_refused_only_by_the_derivative(
    constructor_state,
):
    _, *graphs = constructor_state
    for graph in graphs:
        value = run_named(graph, "solve", (1.0, 0.0, 2.0)).value
        assert isinstance(value, torch.Tensor) and value.item() == 3.0
        with pytest.raises(JvpBoundaryError, match="input-dependent equality"):
            exact_jvp(graph, "solve", (1, 0, 2), (0, 1, 0))
        with pytest.raises(JvpBoundaryError, match="input-dependent equality"):
            exact_gradient(graph, "solve", (1, 0, 2))


def test_constructor_state_oracle_rejects_the_wrong_transferred_state(
    constructor_state,
):
    _, source, direct, _, _ = constructor_state
    (companion,) = [
        item
        for item in direct.computations
        if item.id not in {entry.id for entry in source.computations}
    ]
    assert isinstance(companion.body, Case)
    hits = 0

    def transfer(body):
        nonlocal hits
        if isinstance(body, Bind):
            return body.with_(first=transfer(body.first), then=transfer(body.then))
        if isinstance(body, Call) and body.callee == companion.id:
            hits += 1
            return body.with_(
                arguments=(*body.arguments[:-1], Var(local=companion.parameters[-1]))
            )
        return body

    mutated = _replace_body(
        direct,
        companion,
        companion.body.with_(
            branches=tuple(
                branch.with_(body=transfer(branch.body))
                for branch in companion.body.branches
            )
        ),
    )
    assert hits == 1
    validate_module(mutated)
    point = _CASE_INPUTS[1]
    live, calls, scores, _, _ = _constructor_trace(
        source, direct, point, _scoring_runtime()
    )
    changed, changed_calls, changed_scores, _, _ = _constructor_trace(
        source, mutated, point, _scoring_runtime()
    )
    assert changed_calls == calls and changed_scores == scores
    assert changed == (11.0, 11.0) != live
    gradient = exact_gradient(mutated, "solve", point)
    assert (gradient.value, gradient.gradient) == (11, (11, 0, 0))
    assert gradient.gradient != _constructor_polynomial(*point)[2]


def test_constructor_state_companion_size_is_independent_of_the_runtime_spine(
    constructor_state,
):
    _, source, _, _, _ = constructor_state
    entry = _entry(source, "from_path")
    direct = _specialize(source, "from_path")
    prior = {item.id for item in source.computations}
    (companion,) = [item for item in direct.computations if item.id not in prior]
    terms = sum(
        sum(1 for _ in _terms(item.body))
        for item in (_entry(direct, "from_path"), companion)
    )
    with pytest.raises(StateSpecializationLimit, match=f"bound of {terms - 1}"):
        specialize_recursive_state(source, entry.id, max_terms=terms - 1)
    assert specialize_recursive_state(source, entry.id, max_terms=terms) == direct

    path = run_named(source, "schedule", (-1.0, 2.0)).value
    for n in range(41):
        if n in (0, 1, 40):
            values = tuple(2 ** (i + 1) - 1 for i in range(n + 1))
            expected = (2 * n + 3) * values[-1] + 2 * sum(values[:-1])
            for graph in (source, direct):
                obtained = run_named(graph, "from_path", (path, 1.0)).value
                assert isinstance(obtained, torch.Tensor)
                assert obtained.item() == expected
        if n < 40:
            path = run_named(source, "prepend", (1.0, 2.0, path)).value
    for graph in (source, direct):
        with pytest.raises(ExecutionFailure) as error:
            run_named(graph, "from_path", (path, 1.0), fuel=16, record_trace=False)
        assert error.value.diagnostic.code == "qiec-run-fuel"


def test_constructor_state_runs_through_both_emitted_pyro_forms(constructor_state):
    compiler, _, _, staged, _ = constructor_state
    report = compiler.effect_plan("solve").check_target(
        "pyro", arguments=_CASE_INPUTS[1], direction=(1, 2, -1)
    )
    assert report.projection == "source" and report.refusals == ()
    assert report.external_derivative.missing == ("external derivative unverified",)
    projection = project_for_target(staged, discharge_module(staged), "pyro")
    plan = IRProgram(name=projection.module, inputs=(), body=(), module=projection)
    code = (
        transpile(parse(_CONSTRUCTOR_STATE), target="pyro"),
        _emit_target(plan, "pyro"),
    )
    for emitted in code:
        namespace = {}
        exec(emitted, namespace)
        for point in _CASE_INPUTS:
            _, expected, gradient = _constructor_polynomial(*point)
            arguments = tuple(
                torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
                for value in point
            )
            value = namespace["qiec_solve"](*arguments)
            assert isinstance(value, torch.Tensor) and value.item() == expected
            measured = torch.autograd.grad(value, arguments, allow_unused=True)
            assert (
                tuple(0 if item is None else item.item() for item in measured)
                == gradient
            )


@pytest.mark.parametrize("static_dependency", ("pattern", "motive"))
def test_state_constructor_static_dependencies_still_refuse(static_dependency):
    source = _CONSTRUCTOR_STATE
    if static_dependency == "pattern":
        source = (
            source.replace("constructor Step :", "constructor Step[tag : Nat] :")
            .replace("construct Step(", "construct Step[1](")
            .replace(
                "Step(amount, scale, rest) =>", "Step[tag](amount, scale, rest) =>"
            )
        )
    else:
        source = (
            source.replace("Path", "Path(0)")
            .replace("family Path(0) : Type", "family Path(position : Nat) : Type")
            .replace("case path motive =>", "case path motive (position : Nat) =>")
        )
    module = Compiler(parse(source), module_name="static-state-case").qiec_module
    assert module is not None
    value = run_named(module, "solve", (1.0, 2.0, 2.0)).value
    assert isinstance(value, torch.Tensor) and value.item() == 122
    with pytest.raises(StateSpecializationUnsupported, match="monomorphic family"):
        _specialize(module, "solve")
    ordinary = discharge_module(module)
    obstacle = ordinary.obstacle("solve")
    assert obstacle is not None and obstacle.kind == "recursion"
    assert _entry(ordinary.module, "solve") == _entry(module, "solve")
    with pytest.raises(JvpUnsupported, match="ConstructorValue"):
        exact_gradient(module, "solve", (1, 2, 2))


def test_nested_installation_and_unrestricted_grade_are_refused(module):
    with pytest.raises(StateSpecializationUnsupported, match="Handle"):
        _specialize(module, "nested_scope")
    handler = next(item for item in module.handlers if item.name == "threaded")
    changed = handler.with_(
        clauses=(
            handler.clauses[0].with_(grade=ResumptionGrade.UNRESTRICTED),
            *handler.clauses[1:],
        )
    )
    candidate = module.with_(
        handlers=tuple(
            changed if item.id == changed.id else item for item in module.handlers
        )
    )
    validate_module(candidate)
    with pytest.raises(StateSpecializationUnsupported, match="linear grade"):
        _specialize(candidate, "execute")
    refused = discharge_module(candidate)
    obstacle = refused.obstacle("execute")
    assert obstacle is not None and obstacle.kind == "recursion"
    assert _entry(refused.module, "execute") is _entry(candidate, "execute")


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
