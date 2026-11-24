"""Static discharge of authored handlers.

A discharged computation installs no handler, has the type of the original, and
has its value at every argument: the reference machine runs both. The random
programs request under handlers of every grade in a random nesting order, with
parameters, with clauses that resume in and out of tail position, and with
requests under conditionals and inside callees.
"""

from __future__ import annotations

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    CheckContext,
    DISCHARGE_STATEMENTS,
    Bind,
    Call,
    Case,
    Computation,
    DischargeObstacle,
    Handle,
    If,
    NamedComputation,
    NewInstance,
    Perform,
    QiecModule,
    Resume,
    Return,
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
    return computation


def _agree(module: QiecModule, name: str, arguments: tuple[object, ...]) -> object:
    """A computation's value, the same before and after discharge."""
    _discharged(module, name)
    original = run_named(module, name, arguments)
    rewritten = run_named(discharge_module(module).module, name, arguments)
    assert rewritten.value == pytest.approx(original.value, rel=1e-9, abs=1e-12)
    assert not any(event.event.startswith("handler") for event in rewritten.trace)
    assert any(event.event == "handler.entered" for event in original.trace)
    return original.value


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

define score(out : Outcome) : Real !{ab} =
    case out motive => Real
        Won(x) =>
            return x
        Lost =>
            let y <- perform ab.stop(0.0)
            return y

define subject(out : Outcome) : Real !{} =
    handle ab with stopped in
        let x <- score(out)
        return x + 1.0
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
    assert (
        infer_computation(pair.body, registry, context).result == pair.type.result
    )
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


def test_a_module_without_handlers_is_returned_as_it_is() -> None:
    module = _module(
        """
define bump(x : Real) : Real !{} =
    return x + 1.0
"""
    )
    assert discharge_module(module).module is module
