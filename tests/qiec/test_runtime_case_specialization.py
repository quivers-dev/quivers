"""Selected cases specialize the actual constructor data by rigid identity."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import run_named
from quivers.qiec.module import QiecModule
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import Bind, Case
from quivers.qiec.types import IndexLiteral, IndexVariable, TypeApplication


SOURCE = """\
family Measurement(position : Nat) : Type
    constructor measured[n : Nat] : Real -> Measurement(n)

family Envelope[A : Type](position : Nat) : Type
    constructor packed[visible : Nat, hidden : Nat] : A * Measurement(hidden) -> Envelope[A](visible)

family Repacked(position : Nat) : Type
    constructor stored[visible : Nat, hidden : Nat] : Measurement(hidden) -> Repacked(visible)

family Paired(first : Nat, second : Nat) : Type
    constructor paired[i : Nat, j : Nat] : Repacked(i) * Repacked(j) -> Paired(i, j)

define nested(left : Envelope[Real](1), right : Envelope[Bool](2)) : Paired(1, 2) !{} =
    case left motive (first : Nat) => Paired(first, 2)
        packed[a, h](unused, read) =>
            let saved = construct stored[a, h](read) as Repacked(a)
            case right motive (second : Nat) => Paired(a, second)
                packed[b, h](keep, other) =>
                    let found = construct stored[b, h](other) as Repacked(b)
                    return construct paired[a, b](saved, found) as Paired(a, b)

define paired_records(offset : Real) : Paired(1, 2) !{} =
    let number = construct measured[3](1.25 + offset) as Measurement(3)
    let other = construct measured[7](4.75) as Measurement(7)
    let first = construct packed[Real, 1, 3](0.0, number) as Envelope[Real](1)
    let second = construct packed[Bool, 2, 7](true, other) as Envelope[Bool](2)
    nested(first, second)
"""


def test_nested_cases_specialize_disjoint_scopes_from_actual_arguments() -> None:
    module = load_source(SOURCE).qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    nested = next(item.body for item in restored.computations if item.name == "nested")
    assert isinstance(nested, Case)
    (outer,) = nested.branches
    assert isinstance(outer.body, Bind) and isinstance(outer.body.then, Case)
    (inner,) = outer.body.then.branches
    outer_hidden = outer.static_arguments[1]
    inner_hidden = inner.static_arguments[1]
    assert isinstance(outer_hidden, IndexVariable) and isinstance(
        inner_hidden, IndexVariable
    )
    assert outer_hidden.name == inner_hidden.name
    assert outer_hidden.identity != inner_hidden.identity
    offset = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    pair = run_named(restored, "paired_records", (offset,)).value
    assert isinstance(pair, RuntimeConstructor)
    data = []
    for record, visible, hidden, weight in zip(
        pair.fields, (1, 2), (3, 7), (1.75, 4.75), strict=True
    ):
        assert isinstance(record, RuntimeConstructor)
        assert isinstance(record.result_type, TypeApplication)
        (result,) = record.result_type.arguments
        assert isinstance(result, IndexLiteral) and result.value == visible
        assert all(isinstance(arg, IndexLiteral) for arg in record.static_arguments)
        assert tuple(arg.value for arg in record.static_arguments) == (visible, hidden)
        (observation,) = record.fields
        assert isinstance(observation, RuntimeConstructor)
        (datum,) = observation.fields
        assert isinstance(datum, torch.Tensor)
        torch.testing.assert_close(datum, offset.new_tensor(weight))
        data.append(datum)
    (gradient,) = torch.autograd.grad(sum(data), offset)
    torch.testing.assert_close(gradient, offset.new_ones(()))


def test_specialization_never_repairs_a_mistyped_checked_result() -> None:
    source = SOURCE.replace(
        "define paired_records(offset : Real) : Paired(1, 2)",
        "define paired_records(offset : Real) : Paired(1, 1)",
    )
    with pytest.raises(
        QiecDiagnosticError, match=r"returns Paired\[1, 2\].*Paired\[1, 1\]"
    ) as error:
        load_source(source)
    assert error.value.code == "qiec-kind"
