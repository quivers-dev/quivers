"""Handlers with parameters: syntax, typing, execution, and serialization.

A handler's parameters are typed binders given where the handler is installed.
Every clause reads them, a ``resume`` names the values they have while the
continuation runs, and a ``resume`` that names none carries them on.
"""

from __future__ import annotations

import pytest

from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.ast_nodes import QiecHandlerDecl
from quivers.dsl.program_theory import extract_program_schema
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import qiec_local_bindings
from quivers.qiec import (
    ExecutionFailure,
    KernelError,
    NewInstance,
    Handle,
    QiecModule,
    Resume,
    dumps,
    loads,
    run_named,
    validate_module,
)

STATE = """\
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit
"""

COUNTED = (
    STATE
    + """
handler cell[S : Type, A : Type] for State[S] : A -> A with (s : S) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(s) with (s)
    put(v : S) resumes 1 =>
        resume(unit) with (v)

define counted(seed : Int) : Int !{} =
    with instance c : State[Int] in
        handle c with cell[Int, Int](seed) in
            let before <- perform c.get()
            perform c.put(before + 1)
            let after <- perform c.get()
            return after
"""
)


def _compiler(source: str) -> Compiler:
    """A compiled source."""
    compiler = Compiler(parse(source), module_name="params", file_path="p.qvr")
    compiler.compile()
    return compiler


def _module(source: str) -> QiecModule:
    """A source's checked module."""
    module = _compiler(source).qiec_module
    assert module is not None
    return module


def _run(source: str, name: str, *arguments: object) -> object:
    """A computation's value on the reference machine."""
    return run_named(_module(source), name, arguments).value


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_a_state_cell_counts() -> None:
    assert _run(COUNTED, "counted", 3) == 4


def test_a_resume_without_with_keeps_the_parameters() -> None:
    assert _run(COUNTED.replace("resume(s) with (s)", "resume(s)"), "counted", 3) == 4
    # The update is what carries the count: a `put` that keeps the parameter
    # loses it.
    lost = COUNTED.replace("resume(unit) with (v)", "resume(unit)")
    assert _run(lost, "counted", 3) == 3


def test_the_return_clause_reads_the_final_state() -> None:
    source = (
        STATE
        + """
handler final for State[Int] : Int -> Int * Int with (s : Int) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, s)
    get resumes 1 =>
        resume(s)
    put(v : Int) resumes 1 =>
        resume() with (v)

define stepped(seed : Int) : Int * Int !{} =
    with instance c : State[Int] in
        handle c with final(seed) in
            let before <- perform c.get()
            perform c.put(before * 5)
            return before
"""
    )
    assert _run(source, "stepped", 3) == (3, 15)


def test_a_handler_that_never_updates_its_parameter_is_a_reader() -> None:
    source = """
effect Reader[R : Type]
    ask : Unit -> R

handler read[R : Type, A : Type] for Reader[R] : A -> A with (r : R) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    ask resumes 1 =>
        resume(r)

define twice(r : Int) : Int !{} =
    with instance e : Reader[Int] in
        handle e with read[Int, Int](r) in
            let a <- perform e.ask()
            let b <- perform e.ask()
            return a + b
"""
    assert _run(source, "twice", 21) == 42


def test_a_handler_whose_return_clause_answers_its_parameter_is_a_writer() -> None:
    source = """
effect Writer
    tell : Int -> Unit

handler collect[A : Type] for Writer : A -> A * Int with (written : Int) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, written)
    tell(v : Int) resumes 1 =>
        resume() with (written + v)

define told(a : Int) : Bool * Int !{} =
    with instance w : Writer in
        handle w with collect[Bool](0) in
            perform w.tell(a)
            perform w.tell(a * 10)
            return true
"""
    assert _run(source, "told", 2) == (True, 22)


def test_two_parameters_are_updated_in_order() -> None:
    source = """
effect Step
    step : Unit -> Int

handler fib for Step : Int -> Int with (a : Int, b : Int) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    step resumes 1 =>
        resume(a) with (b, a + b)

define digits() : Int !{} =
    with instance s : Step in
        handle s with fib(0, 1) in
            let x0 <- perform s.step()
            let x1 <- perform s.step()
            let x2 <- perform s.step()
            let x3 <- perform s.step()
            let x4 <- perform s.step()
            let x5 <- perform s.step()
            return x0 + x1 * 10 + x2 * 100 + x3 * 1000 + x4 * 10000 + x5 * 100000
"""
    assert _run(source, "digits") == 532110


NESTED = COUNTED.replace(
    "define counted",
    """define crossed(a : Int, b : Int) : Int !{} =
    with instance left : State[Int] in
        handle left with cell[Int, Int](a) in
            with instance right : State[Int] in
                handle right with cell[Int, Int](b) in
                    let x <- perform left.get()
                    let y <- perform right.get()
                    perform left.put(y)
                    perform right.put(x * 100)
                    let p <- perform left.get()
                    let q <- perform right.get()
                    return p + q

define counted""",
)


def test_nested_handlers_of_one_interface_keep_their_own_state() -> None:
    assert _run(NESTED, "crossed", 3, 7) == 307


SHOTS = (
    STATE
    + """
effect Flip
    flip : Unit -> Bool

handler cell for State[Int] : Int -> Int with (s : Int) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x * 1000 + s
    get resumes 1 =>
        resume(s)
    put(v : Int) resumes 1 =>
        resume() with (v)

handler numbered for Flip : Int -> Int with (n : Int) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x + n
    flip resumes omega =>
        let heads <- resume(true) with (n + 1)
        let tails <- resume(false) with (n + 10)
        return heads * 7 + tails + n * 100

define step(n : Int, b : Bool) : Int !{} =
    if b then
        return n + 1
    else
        return n + 2

define inner_state() : Int !{} =
    with instance f : Flip in
        handle f with numbered(0) in
            with instance c : State[Int] in
                handle c with cell(5) in
                    let b <- perform f.flip()
                    let s <- perform c.get()
                    let t <- step(s, b)
                    perform c.put(t)
                    return 1

define outer_state() : Int !{} =
    with instance c : State[Int] in
        handle c with cell(5) in
            with instance f : Flip in
                handle f with numbered(0) in
                    let b <- perform f.flip()
                    let s <- perform c.get()
                    let t <- step(s, b)
                    perform c.put(t)
                    return t
"""
)


def test_each_shot_has_the_parameters_its_resume_names() -> None:
    # Each shot restarts from the captured state 5, and the return clauses add
    # the parameter that shot's resume named.
    heads = 1 * 1000 + 6 + 1
    tails = 1 * 1000 + 7 + 10
    assert _run(SHOTS, "inner_state") == heads * 7 + tails


def test_a_state_handler_outside_the_choice_is_shared_by_the_shots() -> None:
    # The first shot leaves 6 in the outer cell, which the second shot reads.
    heads = 6 + 1
    tails = 8 + 10
    assert _run(SHOTS, "outer_state") == (heads * 7 + tails) * 1000 + 8


def test_traverse_resumes_once_per_item() -> None:
    source = """
effect Pick
    pick : Unit -> Real

handler each for Pick : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    pick resumes omega =>
        let options = [1.0, 2.0, 4.0]
        let answers <- traverse(options, o -> resume(o))
        return sum(answers)

define squares() : Real !{} =
    with instance p : Pick in
        handle p with each in
            let x <- perform p.pick()
            return x * x
"""
    assert _run(source, "squares") == 21.0
    with pytest.raises(QiecDiagnosticError, match="grade"):
        _module(source.replace("resumes omega", "resumes 1"))


# ---------------------------------------------------------------------------
# Typing
# ---------------------------------------------------------------------------

_MISUSES = {
    "a value of another type at the installation": (
        "with cell[Int, Int](seed)",
        "with cell[Int, Int](true)",
        "parameter 0 of handler 'cell' has type Int, got Bool",
    ),
    "no value at the installation": (
        "with cell[Int, Int](seed)",
        "with cell[Int, Int]",
        "handler 'cell' declares 1 parameter(s), got 0 value(s)",
    ),
    "too many values at the installation": (
        "with cell[Int, Int](seed)",
        "with cell[Int, Int](seed, seed)",
        "handler 'cell' declares 1 parameter(s), got 2 value(s)",
    ),
    "a value of another type at a resumption": (
        "resume(s) with (s)",
        "resume(s) with (1)",
        "parameter 0 of the enclosing handler has type S, got Int",
    ),
    "too many values at a resumption": (
        "resume(s) with (s)",
        "resume(s) with (s, s)",
        "the enclosing handler declares 1 parameter(s), got 2 value(s)",
    ),
    "a clause binder that reuses a parameter's name": (
        "put(v : S) resumes 1 =>\n        resume(unit) with (v)",
        "put(s : S) resumes 1 =>\n        resume(unit) with (s)",
        "local already bound: 's'",
    ),
    "two parameters of one name": (
        "with (s : S)",
        "with (s : S, s : S)",
        "local already bound: 's'",
    ),
    "a resumption outside a clause": (
        "            return after",
        "            resume(after)",
        "resume outside a handler clause body",
    ),
}


@pytest.mark.parametrize("misuse", sorted(_MISUSES))
def test_a_misused_parameter_is_rejected(misuse: str) -> None:
    old, new, message = _MISUSES[misuse]
    assert old in COUNTED
    with pytest.raises(QiecDiagnosticError) as raised:
        _module(COUNTED.replace(old, new))
    assert message in str(raised.value)


def test_values_given_to_a_handler_without_parameters_are_rejected() -> None:
    source = (
        STATE
        + """
handler fixed for State[Int] : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(7)
    put(v : Int) resumes 1 =>
        resume()

define counted(seed : Int) : Int !{} =
    with instance c : State[Int] in
        handle c with fixed(seed) in
            let before <- perform c.get()
            return before
"""
    )
    with pytest.raises(QiecDiagnosticError, match="declares 0 parameter"):
        _module(source)


def test_a_foreign_handler_declares_no_parameters() -> None:
    with pytest.raises(QiecDiagnosticError, match="declares parameters"):
        _module(
            STATE
            + """
handler cell for State[Int] : Int -> Int with (s : Int) [coverage=total, forwards=none, implementation=foreign]
    get resumes 1
    put(v : Int) resumes 1
"""
        )


def test_a_clause_performs_only_what_its_handler_introduces() -> None:
    source = (
        COUNTED
        + """
instance outer : State[Int]

handler leaking for State[Int] : Int -> Int with (s : Int) [coverage=total, forwards=none, introduces=!{outer}, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        perform outer.put(s)
        resume(s)
    put(v : Int) resumes 1 =>
        resume() with (v)
"""
    )
    _module(source)
    with pytest.raises(QiecDiagnosticError, match="introduces"):
        _module(source.replace(", introduces=!{outer}", ""))


def test_the_kernel_rechecks_parameters_independently() -> None:
    """A module whose installation or resumption drops a value is refused."""
    module = _module(COUNTED)
    (handler,) = module.handlers
    get, put = handler.clauses
    assert isinstance(put.body, Resume)
    dropped = put.with_(body=put.body.with_(arguments=()))
    with pytest.raises(KernelError, match="resume continues with 0 parameter"):
        validate_module(
            module.with_(handlers=(handler.with_(clauses=(get, dropped)),))
        )
    retyped = put.with_(body=put.body.with_(arguments=(put.body.value,)))
    with pytest.raises(KernelError, match="resume continues with parameter 0"):
        validate_module(
            module.with_(handlers=(handler.with_(clauses=(get, retyped)),))
        )
    counted = next(item for item in module.computations if item.name == "counted")
    assert isinstance(counted.body, NewInstance)
    assert isinstance(counted.body.body, Handle)
    bare = counted.with_(
        body=counted.body.with_(body=counted.body.body.with_(arguments=()))
    )
    with pytest.raises(KernelError, match="is installed with 0 parameter"):
        validate_module(module.with_(computations=(bare,)))


def test_the_reference_machine_checks_an_argument_value() -> None:
    with pytest.raises(ExecutionFailure):
        _run(COUNTED, "counted", "three")


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_parameters_survive_serialization() -> None:
    module = _module(COUNTED)
    text = dumps(module)
    assert loads(text) == module
    assert [item.name for item in module.handlers[0].parameters] == ["s"]
    dropped = module.handlers[0].with_(parameters=())
    assert dumps(module.with_(handlers=(dropped,))) != text


def test_parameters_survive_emission() -> None:
    parsed = parse(NESTED)
    source = module_to_source(parsed)
    assert "with (s : S) [coverage=total" in source
    assert "resume(unit) with (v)" in source
    assert "with cell[Int, Int](seed) in" in source
    assert module_to_source(parse(source)) == source
    assert _run(source, "crossed", 3, 7) == 307


def test_the_multi_line_forms_parse() -> None:
    source = COUNTED.replace(
        "resume(unit) with (v)", "resume(unit) with (\n            v,\n        )"
    )
    assert _run(source, "counted", 3) == 4


def test_a_parameter_is_in_scope_in_every_clause() -> None:
    module = parse(COUNTED)
    declaration = next(
        item for item in module.statements if isinstance(item, QiecHandlerDecl)
    )
    binding = next(
        item
        for item in qiec_local_bindings(module)
        if item.name == "s" and isinstance(item.owner, QiecHandlerDecl)
    )
    assert binding.scope is not None
    first, last = binding.scope
    assert all(first <= clause.line <= last for clause in declaration.clauses)
    assert declaration.line < first


def test_parameters_are_vertices_of_the_program_schema() -> None:
    schema = extract_program_schema(_compiler(COUNTED))
    parameters = [
        vertex for vertex in schema.vertices if vertex.kind == "handler_parameter"
    ]
    assert len(parameters) == 1
    values = {
        constraint.sort: constraint.value
        for constraint in schema.constraints_for(parameters[0].id)
    }
    assert values["name"] == "s"
    assert values["type"] == "S"
