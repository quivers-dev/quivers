"""Checked collection expressions across lowering, execution, and effects."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import parse
from quivers.dsl.ast_nodes import QiecComputationDecl, QiecReturnComputation
from quivers.dsl.compiler.programs import _ProgramsMixin
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec import (
    Bind,
    Call,
    CheckContext,
    Return,
    TensorValue,
    from_data,
    infer_computation,
    to_data,
    validate_module,
)
from quivers.qiec.execution import run_named


def _module(source: str):
    return lower_qvr_to_qiec(parse(source), module_name="collections")


def test_map_named_lambda_lowers_and_runs() -> None:
    module = _module(
        """define mapped() : Tensor[Real]([3]) !{} =
    let twice = x -> x * 2.0
    let ys = map([1.0, 2.0, 3.0], twice)
    return ys
"""
    )
    assert run_named(module, "mapped", ()).value == (2.0, 4.0, 6.0)


def test_map_module_round_trips_through_the_qiec_wire_format() -> None:
    module = _module(
        """define mapped(xs : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return map(xs, x -> x * 2.0)
"""
    )
    assert from_data(to_data(module)) == module


def test_map_is_observationally_equivalent_to_factor() -> None:
    module = _module(
        """object K : FinSet 3

define via_map(xs : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return map(xs, x -> x * 2.0)

define via_factor(xs : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    return factor i : K in xs[i] * 2.0
"""
    )
    values = (1.5, -2.0, 4.0)
    mapped = run_named(module, "via_map", (values,)).value
    factored = run_named(module, "via_factor", (values,)).value
    assert mapped == factored == (3.0, -4.0, 8.0)


def test_named_map_lambda_captures_and_shadows_lexically() -> None:
    module = _module(
        """define captured(xs : Tensor[Real]([2]), offset : Real) : Tensor[Real]([2]) !{} =
    let shift = x -> x + offset
    return map(xs, shift)

define shadowed(xs : Tensor[Real]([2]), x : Real) : Tensor[Real]([2]) !{} =
    let twice = x -> x * 2.0
    return map(xs, twice)
"""
    )
    assert run_named(module, "captured", ((1.0, 2.0), 3.0)).value == (4.0, 5.0)
    assert run_named(module, "shadowed", ((1.0, 2.0), 100.0)).value == (2.0, 4.0)


def test_map_preserves_trailing_shape() -> None:
    module = _module(
        """define rows() : Tensor[Real]([2, 2]) !{} =
    return map([[1.0, 2.0], [3.0, 4.0]], row -> row * 2.0)
"""
    )
    assert run_named(module, "rows", ()).value == ((2.0, 4.0), (6.0, 8.0))


def test_fold_uses_accumulator_then_item() -> None:
    module = _module(
        """define folded() : Real !{} =
    return fold([1.0, 2.0, 3.0], 0.0, acc -> item -> acc + item)
"""
    )
    assert run_named(module, "folded", ()).value == 6.0


def test_length_and_logsumexp_over_run() -> None:
    module = _module(
        """define count() : Int !{} =
    return length([3.0, 4.0, 5.0])

define normalizer() : Real !{} =
    return logsumexp_over([0.0, 1.0, 2.0], x -> x)
"""
    )
    assert run_named(module, "count", ()).value == 3
    assert run_named(module, "normalizer", ()).value == pytest.approx(
        math.log(sum(math.exp(x) for x in (0.0, 1.0, 2.0)))
    )


@pytest.mark.parametrize("extent", (1, 2, 4))
def test_length_returns_an_int_for_fixed_leading_extents(extent: int) -> None:
    values = tuple(float(position) for position in range(extent))
    module = _module(
        f"""define count(xs : Tensor[Real]([{extent}])) : Int !{{}} =
    return length(xs)
"""
    )
    result = run_named(module, "count", (values,)).value
    assert result == extent
    assert isinstance(result, int)


def test_fold_argument_order_agrees_with_the_eager_evaluator() -> None:
    source = """define folded() : Real !{} =
    return fold([1.0, 2.0, 3.0], 0.0, acc -> item -> acc + item)
"""
    parsed = parse(source)
    declaration = parsed.statements[0]
    assert isinstance(declaration, QiecComputationDecl)
    assert isinstance(declaration.body, QiecReturnComputation)
    eager = _ProgramsMixin._compile_let_expr(declaration.body.value)({})
    checked = run_named(_module(source), "folded", ()).value

    assert isinstance(eager, torch.Tensor)
    assert eager.item() == checked == 6.0


def test_logsumexp_over_is_stable_on_extreme_inputs() -> None:
    module = _module(
        """define stable() : Real !{} =
    return logsumexp_over([-1000.0, 0.0, 1000.0], x -> x)

define stable_via_map() : Real !{} =
    let values = map([-1000.0, 0.0, 1000.0], x -> x)
    return logsumexp(values)
"""
    )
    direct = run_named(module, "stable", ()).value
    derived = run_named(module, "stable_via_map", ()).value
    assert direct == pytest.approx(1000.0)
    assert direct == derived


def test_traverse_sequences_calls_and_returns_a_tensor() -> None:
    module = _module(
        """define twice(x : Real) : Real !{} =
    return x * 2.0

define all_twice() : Tensor[Real]([3]) !{} =
    let ys <- traverse([1.0, 2.0, 3.0], x -> twice(x))
    return ys
"""
    )
    assert run_named(module, "all_twice", ()).value == (2.0, 4.0, 6.0)
    computation = next(item for item in module.computations if item.name == "all_twice")
    assert isinstance(computation.body, Bind)
    # The literal source is bound once, and every call reads an item of it.
    source = computation.body.first
    assert isinstance(source, Bind)
    assert isinstance(source.first, Return)
    assert isinstance(source.first.value, TensorValue)
    traversal = source.then
    assert isinstance(traversal, Bind)
    assert isinstance(traversal.first, Call)
    second = traversal.then
    assert isinstance(second, Bind)
    third = second.then
    assert isinstance(third, Bind)
    tail = third.then
    assert isinstance(tail, Return)
    assert isinstance(tail.value, TensorValue)


def test_traverse_joins_the_called_computation_effects() -> None:
    module = _module(
        """instance score : Score

define account(x : Real) : Real !{score} =
    perform score.add(weight(x))
    return x

define account_all() : Tensor[Real]([2]) !{score} =
    let xs <- traverse([1.0, 2.0], x -> account(x))
    return xs
"""
    )
    computation = next(
        item for item in module.computations if item.name == "account_all"
    )
    inferred = infer_computation(
        computation.body,
        validate_module(module),
        CheckContext(locals=computation.parameters),
    )
    assert len(inferred.effects.entries) == 1


def test_traverse_call_sites_are_stable_and_distinct_by_iteration() -> None:
    source = """define twice(x : Real) : Real !{} =
    return x * 2.0

define all_twice() : Tensor[Real]([3]) !{} =
    let ys <- traverse([1.0, 2.0, 3.0], x -> twice(x))
    return ys
"""

    def call_sites():
        module = _module(source)
        computation = next(
            item for item in module.computations if item.name == "all_twice"
        )
        sites = []
        assert isinstance(computation.body, Bind)
        body = computation.body.first
        while isinstance(body, Bind):
            if isinstance(body.first, Call):
                sites.append(body.first.origin.site_id())
            body = body.then
        return tuple(sites)

    first = call_sites()
    assert len(first) == 3
    assert len(set(first)) == 3
    assert call_sites() == first


def test_filter_reports_the_dynamic_shape_boundary() -> None:
    with pytest.raises(
        QiecDiagnosticError, match="data-dependent output length"
    ) as captured:
        _module(
            """define positives() : Tensor[Real]([2]) !{} =
    return filter([-1.0, 1.0], x -> x > 0.0)
"""
        )
    assert captured.value.code == "qiec-collection:filter:dynamic-shape"


def test_collection_delimiters_accept_hanging_indents() -> None:
    module = _module(
        """define hanging() : Tensor[Real]([2]) !{} =
    return map(
        [
            1.0,
            2.0,
        ],
        (x -> x * 2.0),
    )
"""
    )
    assert run_named(module, "hanging", ()).value == (2.0, 4.0)
