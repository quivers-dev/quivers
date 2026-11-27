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
    Handle,
    If,
    NamedComputation,
    NewInstance,
    Perform,
    QiecModule,
    Resume,
    Return,
    RuntimeConfiguration,
    RuntimeSelection,
    discharge_computation,
    discharge_module,
    infer_computation,
    run_named,
    validate_module,
)
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
        elif isinstance(term, NewInstance):
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
    assert isinstance(pair.body, Return)
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


def test_the_bound_holds_of_the_body_and_not_of_the_steps_taken() -> None:
    """Copies of a conditional's branches count against the bound.

    Seed 23 reduces in few steps to a body whose conditionals, nested and
    each stated once per value their branches differ in, pass the bound.
    """
    module = _module(HANDLERS + random_body(23))
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
