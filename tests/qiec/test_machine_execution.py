"""Calls, authored handlers, and local instances on the reference machine.

These pin the machine semantics the checker's new terms depend on: calls
run as frames rather than host recursion, authored clauses resume by
pushing their captured continuation rather than re-entering the evaluator,
local instances address their requests distinctly, and a diverging
computation ends at its fuel rather than at the host's recursion limit.
"""

from __future__ import annotations

import sys
from typing import cast

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    INT,
    ExecutionFailure,
    RuntimeConfiguration,
    RuntimeSelection,
    parse_static_arguments,
    run_named,
)
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.module import QiecModule
from quivers.qiec.types import TypeApplication

type _Address = tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]

_LIST = """family List[A : Type] : Type
    constructor Nil : List[A]
    constructor Cons : A * List[A] -> List[A]
"""


def _module(source: str) -> QiecModule:
    """Compile a QVR source to its checked kernel module.

    Parameters
    ----------
    source : str
        QVR text using the QIEC surface.

    Returns
    -------
    QiecModule
        The lowered and independently validated module.
    """
    compiler = Compiler(parse(source), module_name="machine", file_path="machine.qvr")
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _int_list(module: QiecModule, items: list[int]) -> RuntimeConstructor:
    """Build the runtime value of a ``List[Int]`` from Python integers.

    Parameters
    ----------
    module : QiecModule
        A module declaring the ``List`` family above.
    items : list[int]
        The elements, first to last.

    Returns
    -------
    RuntimeConstructor
        The erased constructor chain the evaluator matches on.
    """
    constructors = {item.name: item for item in module.constructors}
    family = next(item for item in module.families if item.name == "List")
    type_ = TypeApplication(constructor=family.type_constructor, arguments=(INT,))
    value = RuntimeConstructor(
        constructor=constructors["Nil"].id,
        static_arguments=(),
        fields=(),
        result_type=type_,
    )
    for item in reversed(items):
        value = RuntimeConstructor(
            constructor=constructors["Cons"].id,
            static_arguments=(),
            fields=(item, value),
            result_type=type_,
        )
    return value


def test_direct_recursion_runs_as_frames_not_host_recursion() -> None:
    """A recursion fifty thousand deep finishes under a tiny host limit."""
    module = _module(
        _LIST
        + """
define last[A : Type](xs : List[A], fallback : A) : A !{} =
    case xs motive => A
        Nil =>
            return fallback
        Cons(head, tail) =>
            last[A](tail, head)
"""
    )
    statics = parse_static_arguments(module, "last", ("A=Int",))
    depth = 50_000
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(400)
    try:
        result = run_named(
            module,
            "last",
            (_int_list(module, list(range(depth))), -1),
            static_arguments=statics,
        )
    finally:
        sys.setrecursionlimit(previous)
    assert result.value == depth - 1
    entered = [event for event in result.trace if event.event == "call.entered"]
    assert len(entered) == depth
    # Every tail call retires the caller's frame, so the trace records each
    # return as a tail return rather than an unwinding one.
    tails = [
        event
        for event in result.trace
        if event.event == "call.returned" and event.detail.get("tail") is True
    ]
    assert len(tails) == depth - 1


def test_non_tail_recursion_also_runs_as_frames() -> None:
    """Binding a recursive result keeps the caller's frame on the machine stack."""
    module = _module(
        _LIST
        + """
define first[A : Type](xs : List[A], fallback : A) : A !{} =
    case xs motive => A
        Nil =>
            return fallback
        Cons(head, tail) =>
            let rest <- first[A](tail, head)
            return head
"""
    )
    statics = parse_static_arguments(module, "first", ("A=Int",))
    depth = 20_000
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(400)
    try:
        result = run_named(
            module,
            "first",
            (_int_list(module, list(range(depth))), -1),
            static_arguments=statics,
        )
    finally:
        sys.setrecursionlimit(previous)
    assert result.value == 0
    returned = [event for event in result.trace if event.event == "call.returned"]
    assert len(returned) == depth
    assert not any(event.detail.get("tail") for event in returned)


def test_mutual_recursion_alternates_between_computations() -> None:
    """Two computations calling each other resolve by identity at each step."""
    module = _module(
        _LIST
        + """
define even_length[A : Type](xs : List[A]) : Bool !{} =
    case xs motive => Bool
        Nil =>
            return true
        Cons(head, tail) =>
            odd_length[A](tail)

define odd_length[A : Type](xs : List[A]) : Bool !{} =
    case xs motive => Bool
        Nil =>
            return false
        Cons(head, tail) =>
            even_length[A](tail)
"""
    )
    statics = parse_static_arguments(module, "even_length", ("A=Int",))
    for length in (0, 1, 2, 7, 10):
        result = run_named(
            module,
            "even_length",
            (_int_list(module, list(range(length))),),
            static_arguments=statics,
        )
        assert result.value is (length % 2 == 0)
        names = [
            event.detail["computation"]
            for event in result.trace
            if event.event == "call.entered"
        ]
        assert names == (["odd_length", "even_length"] * length)[:length]


def test_fuel_exhaustion_is_a_stable_diagnostic() -> None:
    """A diverging computation ends with ``qiec-run-fuel``, not a host error."""
    module = _module("define spin(seed : Int) : Int !{} =\n    spin(seed)\n")
    with pytest.raises(ExecutionFailure) as error:
        run_named(module, "spin", (1,), fuel=500)
    assert error.value.diagnostic.code == "qiec-run-fuel"
    assert error.value.diagnostic.computation == "spin"
    assert "500" in error.value.diagnostic.message


_STATE = """effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

handler run_state for State[Int] : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(0)
    put(s : Int) resumes 1 =>
        resume(unit)
"""


def test_authored_handler_runs_without_an_attachment() -> None:
    """An authored handler's clauses run from the module, needing no provider."""
    module = _module(
        _STATE
        + """
define counted(seed : Int) : Int !{} =
    with instance cell : State[Int] in
        handle cell with run_state in
            perform cell.put(seed)
            let current <- perform cell.get()
            return current
"""
    )
    result = run_named(module, "counted", (5,))
    assert result.value == 0
    events = [event.event for event in result.trace]
    assert events.index("instance.allocated") < events.index("handler.entered")
    assert events.count("clause.entered") == 2
    assert events.count("resumption.invoked") == 2
    # Deep semantics: the handled computation's return, and so the
    # handler's return clause, happens inside the last clause's resumption,
    # before that clause's answer is recorded.
    assert events.index("handler.returned") < events.index("clause.answered")
    assert events[-2] == "instance.released"


def test_unrestricted_authored_clause_resumes_twice() -> None:
    """A multi-shot authored clause runs its continuation once per resume."""
    module = _module(
        """effect Flip
    flip : Unit -> Bool

instance coin : Flip

handler both for Flip : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    flip resumes omega =>
        let heads <- resume(true)
        let tails <- resume(false)
        return tails

define flipped() : Int !{} =
    handle coin with both in
        let outcome <- perform coin.flip()
        return 1
"""
    )
    result = run_named(module, "flipped")
    assert result.value == 1
    shots = [
        event.detail["shot"]
        for event in result.trace
        if event.event == "resumption.invoked"
    ]
    assert shots == [0, 1]
    assert [e.event for e in result.trace].count("handler.returned") == 2


def test_nested_local_instances_address_requests_distinctly() -> None:
    """Two allocations of one interface give their requests different addresses."""
    module = _module(
        _STATE
        + """
define nested(seed : Int) : Int !{} =
    with instance outer : State[Int] in
        handle outer with run_state in
            with instance inner : State[Int] in
                handle inner with run_state in
                    perform outer.put(seed)
                    perform inner.put(seed)
                    let value <- perform inner.get()
                    return value
"""
    )
    result = run_named(module, "nested", (3,))
    addresses = [
        cast(_Address, event.detail["address"])
        for event in result.trace
        if event.event == "operation.requested"
    ]
    dynamic = [address[1] for address in addresses]
    assert dynamic[0] == (("instance", 1), ("instance", 2))
    assert dynamic[1] == (("instance", 1), ("instance", 2))
    assert len({address[0] for address in addresses}) == 3
    # Identical invocations address identically; the serials restart per run.
    again = run_named(module, "nested", (3,))
    assert [
        cast(_Address, event.detail["address"])
        for event in again.trace
        if event.event == "operation.requested"
    ] == addresses


def test_recursive_iterations_address_requests_distinctly() -> None:
    """One request site reached on each iteration yields a distinct address."""
    module = _module(
        _LIST
        + """
effect Tick
    tick : Unit -> Unit

instance clock : Tick

handler count for Tick : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    tick resumes 1 =>
        resume(unit)

define ticking(xs : List[Int]) : Int !{clock} =
    case xs motive => Int
        Nil =>
            return 0
        Cons(head, tail) =>
            perform clock.tick()
            let rest <- ticking(tail)
            return rest

define run(xs : List[Int]) : Int !{} =
    handle clock with count in
        ticking(xs)
"""
    )
    result = run_named(module, "run", (_int_list(module, [1, 2, 3]),))
    addresses = [
        cast(_Address, event.detail["address"])
        for event in result.trace
        if event.event == "operation.requested"
    ]
    assert len(addresses) == 3
    assert len({address[0] for address in addresses}) == 1
    assert len(set(addresses)) == 3
    assert [address[1] for address in addresses] == [
        (("call", "ticking#1"),),
        (("call", "ticking#1"), ("call", "ticking#2")),
        (("call", "ticking#1"), ("call", "ticking#2"), ("call", "ticking#3")),
    ]


def test_authored_and_foreign_handlers_nest_in_either_order() -> None:
    """An authored handler under a foreign one, and the reverse, both run."""
    source = (
        _STATE
        + """
effect Log
    note : Int -> Unit

handler record for Log : Int -> Int [coverage=total, forwards=none, implementation=foreign]
    note(value : Int) resumes 1

define authored_inside(seed : Int) : Int !{} =
    with instance journal : Log in
        handle journal with record in
            with instance cell : State[Int] in
                handle cell with run_state in
                    perform journal.note(seed)
                    perform cell.put(seed)
                    let value <- perform cell.get()
                    return value

define foreign_inside(seed : Int) : Int !{} =
    with instance cell : State[Int] in
        handle cell with run_state in
            with instance journal : Log in
                handle journal with record in
                    perform cell.put(seed)
                    perform journal.note(seed)
                    let value <- perform cell.get()
                    return value
"""
    )
    module = _module(source)
    runtime = RuntimeConfiguration(
        selections=(
            RuntimeSelection(
                provider="core",
                options={
                    "handlers": {
                        "record": {"kind": "scripted", "responses": {"note": [None]}}
                    }
                },
            ),
        )
    )
    for name in ("authored_inside", "foreign_inside"):
        result = run_named(module, name, (4,), runtime=runtime)
        assert result.value == 0
        handled = [
            cast(str, event.detail["handler"])
            for event in result.trace
            if event.event == "operation.handled"
        ]
        assert sorted(handled) == ["record", "run_state", "run_state"]
