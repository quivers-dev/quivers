"""Collection forms share intermediate values through ``LetValue`` bindings.

A ``fold`` body that reads its accumulator several times, a ``map`` over a
computed source, and a lambda-bound name applied to a computed argument
each bind the shared value once. The lowered term thus grows linearly in
the collection's length, and the reference machine, the torch runtime, and
every host runtime evaluate it to the same number a hand-written loop
computes.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields, is_dataclass

import pytest
import torch

from quivers.dsl import Compiler, loads, parse
from quivers.dsl.ast_nodes import QiecComputationDecl, QiecReturnComputation
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec import LetValue, QiecModule, from_data, to_data
from quivers.qiec.execution import run_named
from quivers.qiec.program_runtime import run_program


def _module(source: str) -> QiecModule:
    return lower_qvr_to_qiec(parse(source), module_name="shared")


def _node_count(term: object) -> int:
    """The number of term records in a lowered computation body."""
    count = 0
    pending: list[object] = [term]
    while pending:
        item = pending.pop()
        if isinstance(item, tuple):
            pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            count += 1
            pending.extend(getattr(item, field.name) for field in fields(item))
    return count


def _body_size(module: QiecModule, name: str) -> int:
    computation = next(item for item in module.computations if item.name == name)
    return _node_count(computation.body)


def _fold_source(length: int, reads: int) -> str:
    body = " + ".join(f"acc * {0.5 / (read + 1)}" for read in range(reads))
    return f"""define folded(xs : Tensor[Real]([{length}])) : Real !{{}} =
    return fold(xs, 0.25, acc -> x -> {body} + x)
"""


def _python_fold(xs: tuple[float, ...], reads: int) -> float:
    acc = 0.25
    for x in xs:
        acc = sum(acc * (0.5 / (read + 1)) for read in range(reads)) + x
    return acc


def _points(length: int) -> tuple[float, ...]:
    return tuple(math.sin(position + 1.0) for position in range(length))


@pytest.mark.parametrize("reads", [1, 2, 3])
def test_a_fold_lowers_to_a_term_linear_in_its_length(reads: int) -> None:
    sizes = {
        length: _body_size(_module(_fold_source(length, reads)), "folded")
        for length in (8, 16, 32, 64)
    }
    # Every step adds the same number of records: doubling the length at
    # most doubles the term. A substituted accumulator would multiply it
    # by the number of reads at every step instead.
    for length in (8, 16, 32):
        assert sizes[2 * length] <= 2 * sizes[length] + 8
    assert sizes[64] < 120 * 64


@pytest.mark.parametrize("reads", [1, 2, 3])
def test_a_fold_agrees_with_a_python_fold_on_the_reference_machine(
    reads: int,
) -> None:
    points = _points(64)
    module = _module(_fold_source(64, reads))
    assert run_named(module, "folded", (points,)).value == pytest.approx(
        _python_fold(points, reads), rel=1e-12
    )
    assert from_data(to_data(module)) == module


def test_a_fold_binds_each_intermediate_accumulator() -> None:
    module = _module(_fold_source(4, 2))
    computation = next(item for item in module.computations if item.name == "folded")
    value = computation.body.value  # type: ignore[union-attr]
    assert isinstance(value, LetValue)
    assert len(value.bindings) == 3


def test_nested_lambda_applications_share_their_argument() -> None:
    source = """define nested(x : Real) : Real !{} =
    let twice = y -> y * 0.5 + y * 0.25
    return twice(twice(twice(twice(twice(twice(twice(twice(x + 1.0))))))))
"""
    module = _module(source)
    expected = 1.5
    for _ in range(8):
        expected = expected * 0.5 + expected * 0.25
    assert run_named(module, "nested", (0.5,)).value == pytest.approx(expected)
    assert _body_size(module, "nested") < 400


def _nested_maps(depth: int) -> str:
    expression = "xs"
    for _ in range(depth):
        expression = f"map({expression}, v -> v * 0.5 + v * 0.25)"
    return f"""define mapped(xs : Tensor[Real]([3])) : Tensor[Real]([3]) !{{}} =
    return {expression}
"""


def test_nested_maps_over_computed_sources_stay_linear() -> None:
    module = _module(_nested_maps(8))
    points = (1.0, -2.0, 3.0)
    assert run_named(module, "mapped", (points,)).value == pytest.approx(
        tuple(x * 0.75**8 for x in points)
    )
    # Each map binds its source once, so a level adds a fixed number of
    # records; substituting the source for each of the two reads of its
    # item would double the term at every level.
    shallow = _body_size(_module(_nested_maps(4)), "mapped")
    assert _body_size(module, "mapped") <= 2 * shallow + 40


PROGRAM = """
object Draw : FinSet 1
object Val : Real 1

program prog : Draw -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let message = fold([{items}], [a, a], m -> x -> map([0, 1], j -> m[0] * 0.5 + m[1] * 0.25 + x))
    let total = fold([{items}], a, acc -> x -> acc * acc * 0.1 + acc * 0.2 + x)
    score s = sum(message) + total
    return a

export prog
"""


def _program_source(length: int) -> str:
    return PROGRAM.format(items=", ".join(f"{0.01 * (k + 1)}" for k in range(length)))


def _program_score(length: int, a: float) -> float:
    items = [0.01 * (k + 1) for k in range(length)]
    message = [a, a]
    total = a
    for x in items:
        message = [message[0] * 0.5 + message[1] * 0.25 + x] * 2
        total = total * total * 0.1 + total * 0.2 + x
    log_prior = -0.5 * a * a - 0.5 * math.log(2.0 * math.pi)
    return log_prior + sum(message) + total


def test_a_long_fold_in_a_program_loads_quickly() -> None:
    start = time.perf_counter()
    loads(_program_source(64))
    # Generous: the load is linear in the length, well under a second
    # on a laptop. The unshared lowering did not finish at this length.
    assert time.perf_counter() - start < 60.0


def test_a_program_fold_scores_alike_on_the_reference_machine_and_torch() -> None:
    length = 12
    source = _program_source(length)
    expected = _program_score(length, 0.3)
    reference = run_program(_module(source), "prog", data={}, sites={"a": 0.3})
    assert reference.log_joint == pytest.approx(expected, rel=1e-9)
    program = Compiler(parse(source)).compile()
    torch.manual_seed(0)
    traced = trace(
        program._morphism,  # type: ignore[attr-defined]
        torch.zeros(1, 1),
        observations={"a": torch.tensor(0.3)},
    ).log_joint
    assert traced is not None
    assert float(traced.sum()) == pytest.approx(expected, rel=1e-5)


STATES = 4
STEPS = 50


def _forward_source() -> str:
    states = ", ".join(str(state) for state in range(STATES))
    return f"""define forward(start : Tensor[Real]([{STATES}]), transition : Tensor[Real]([{STATES}, {STATES}]), emissions : Tensor[Real]([{STEPS}, {STATES}])) : Real !{{}} =
    return logsumexp_over(fold(emissions, start, message -> emission -> map([{states}], j -> logsumexp_over([{states}], i -> message[i] + transition[i, j]) + emission[j])), score -> score)
"""


def _forward_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    start = torch.log_softmax(
        torch.randn(STATES, generator=generator, dtype=torch.float64), 0
    )
    transition = torch.log_softmax(
        torch.randn(STATES, STATES, generator=generator, dtype=torch.float64), 1
    )
    emissions = torch.randn(STEPS, STATES, generator=generator, dtype=torch.float64)
    return start, transition, emissions


def _hand_forward(
    start: torch.Tensor, transition: torch.Tensor, emissions: torch.Tensor
) -> float:
    alpha = [float(value) for value in start]
    for row in emissions.tolist():
        alpha = [
            math.log(
                sum(math.exp(alpha[i] + float(transition[i, j])) for i in range(STATES))
            )
            + row[j]
            for j in range(STATES)
        ]
    return math.log(sum(math.exp(value) for value in alpha))


def test_a_forward_algorithm_fold_loads_and_matches_a_hand_written_one() -> None:
    start_time = time.perf_counter()
    module = _module(_forward_source())
    assert time.perf_counter() - start_time < 60.0
    assert _body_size(module, "forward") < 600 * STEPS
    start, transition, emissions = _forward_inputs()
    expected = _hand_forward(start, transition, emissions)
    value = run_named(
        module,
        "forward",
        (
            tuple(start.tolist()),
            tuple(tuple(row) for row in transition.tolist()),
            tuple(tuple(row) for row in emissions.tolist()),
        ),
    ).value
    assert value == pytest.approx(expected, rel=1e-9)
    declaration = parse(_forward_source()).statements[0]
    assert isinstance(declaration, QiecComputationDecl)
    assert isinstance(declaration.body, QiecReturnComputation)
    eager = _ProgramsMixin._compile_let_expr(declaration.body.value)(
        {"start": start, "transition": transition, "emissions": emissions}
    )
    assert isinstance(eager, torch.Tensor)
    assert float(eager) == pytest.approx(expected, rel=1e-9)
