"""Static call reuse preserves argument, declaration, and runtime separation."""

from __future__ import annotations

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import INT, STRING, parse_static_arguments, run_named
from quivers.qiec.evaluator import _instantiate_computation
from quivers.qiec.module import QiecModule
from quivers.qiec.types import IndexLiteral, NAT


def _module(result: str = "x") -> QiecModule:
    compiler = Compiler(
        parse(
            f"define choose[A : Type](x : A, y : A) : A !{{}} =\n    return {result}\n"
        ),
        module_name="static-reuse",
        file_path="static-reuse.qvr",
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def test_static_instances_share_graphs_but_separate_types() -> None:
    declaration = _module().computations[0]
    integers = _instantiate_computation(declaration, (INT,))
    strings = _instantiate_computation(declaration, (STRING,))
    assert _instantiate_computation(declaration, (INT,)) is integers
    assert integers[0][0].type == INT
    assert strings[0][0].type == STRING
    assert integers[1] != strings[1]


def test_equal_ids_do_not_alias_different_declaration_bodies() -> None:
    left = _module("x").computations[0]
    right = _module("y").computations[0]
    assert left.id == right.id
    assert (
        _instantiate_computation(left, (INT,))[1]
        != _instantiate_computation(right, (INT,))[1]
    )


def test_reused_static_call_still_reads_each_runs_runtime_arguments() -> None:
    module = _module()
    for type_name, values in (("Int", (3, 8)), ("String", ("first", "next"))):
        arguments = parse_static_arguments(module, "choose", (f"A={type_name}",))
        for value in values:
            assert (
                run_named(
                    module, "choose", (value, value), static_arguments=arguments
                ).value
                == value
            )


@pytest.mark.parametrize(
    "arguments", [(), (INT, STRING), (IndexLiteral(value=0, sort=NAT),)]
)
def test_cached_valid_instance_does_not_accept_invalid_static_arguments(
    arguments,
) -> None:
    declaration = _module().computations[0]
    _instantiate_computation(declaration, (INT,))
    with pytest.raises((TypeError, ValueError)):
        _instantiate_computation(declaration, arguments)
