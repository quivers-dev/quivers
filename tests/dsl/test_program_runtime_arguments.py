"""Program applications preserve stages, declared types, and lexical bindings."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import parse
from quivers.dsl.ast_nodes import (
    DrawArgName,
    DrawArgIndex,
    DrawArgNamed,
    DrawArgScalar,
    LetExprVar,
    ProgramDecl,
    SampleStep,
)
from quivers.dsl.program_templates import (
    TemplateError,
    instantiate_program,
    static_program_parameters,
    template_bindings,
)
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec import QiecModule, dumps, loads
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.program_runtime import program_entry, run_program
from quivers.modeling import Model

HEAD = "object Row : FinSet 3\nobject Out : Real 1\n"
TENSOR_SUB = """program sub[A : FinSet](x : Tensor[Real]([A]), s : Real) : Row -> Out
    sample z : A <- Normal(loc=x, scale=s)
    return z
"""


def _module(source: str):
    return lower_qvr_to_qiec(parse(source), file_path="arguments.qvr")


@pytest.mark.parametrize(
    ("returned", "pattern", "result"),
    [
        ("x", "y", 2.0),
        ("(x, x)", "y", (2.0, 2.0)),
        ("(x, x)", "(a, b)", (2.0, 2.0)),
        ("(x, z)", "y", (2.0, 3.0)),
        ("(z, z)", "(a, b)", (3.0, 3.0)),
    ],
)
def test_returned_input_and_duplicate_return_values(
    returned: str, pattern: str, result: object
) -> None:
    output = "(a, b)" if pattern == "(a, b)" else "y"
    source = (
        HEAD
        + f"program sub(x : Real) : Row -> Out\n    let z = x + 1.0\n    return {returned}\n"
        + f"program main : Row -> Out\n    sample {pattern} <- sub(2.0)\n    return {output}\nexport main\n"
    )
    run = run_program(_module(source), "main", data={}, sites={})
    assert run.value == result
    assert run.log_joint == 0.0


def test_typed_tensor_arguments_preserve_shape_and_density() -> None:
    source = (
        HEAD
        + TENSOR_SUB
        + """program main(x : Tensor[Real]([Row])) : Row -> Out
    sample y <- sub[Row](x, 0.5)
    return y
export main
"""
    )
    module = _module(source)
    means = (1.0, 2.0, 3.0)
    values = (1.1, 2.2, 3.3)
    run = run_program(module, "main", data={"x": means}, sites={"y": values})
    expected = sum(
        -math.log(0.5 * math.sqrt(2.0 * math.pi)) - 0.5 * ((y - x) / 0.5) ** 2
        for x, y in zip(means, values, strict=True)
    )
    assert run.value == values
    assert run.log_joint == pytest.approx(expected)
    assert [
        axis.size.value for axis in program_entry(module, "main").sites[0].batch
    ] == [3]


def test_free_actual_gets_the_formals_declared_tensor_type() -> None:
    source = (
        HEAD
        + TENSOR_SUB
        + """program main : Row -> Out
    sample y <- sub[Row](features, 0.5)
    return y
export main
"""
    )
    module = _module(source)
    entry = program_entry(module, "main")
    assert [(item.name, item.role) for item in entry.parameters] == [
        ("features", "data")
    ]
    run = run_program(
        module, "main", data={"features": (1.0, 2.0, 3.0)}, sites={"y": (1.0, 2.0, 3.0)}
    )
    assert run.value == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("header", "call", "fragment"),
    [
        ("x : Real", "sub[Row](x, 0.5)", "runtime parameter 'x'"),
        ("x : Tensor[Int]([Row])", "sub[Row](x, 0.5)", "runtime parameter 'x'"),
        ("x : Tensor[Real]([2])", "sub[Row](x, 0.5)", "runtime parameter 'x'"),
        ("x : Tensor[Real]([Row])", "sub(x, 0.5)", "static argument"),
        ("x : Tensor[Real]([Row])", "sub[Row](x)", "runtime argument"),
        ("x : Tensor[Real]([Row])", "sub[Row, Row](x, 0.5)", "static argument"),
        (
            "x : Tensor[Real]([Row])",
            "sub[Row](Row, 0.5)",
            "static and cannot fill runtime",
        ),
    ],
)
def test_application_rejects_stage_arity_shape_and_element_mismatches(
    header: str, call: str, fragment: str
) -> None:
    source = (
        HEAD
        + TENSOR_SUB
        + f"program main({header}) : Row -> Out\n    sample y <- {call}\n    return y\nexport main\n"
    )
    with pytest.raises(QiecDiagnosticError, match=fragment):
        _module(source)


@pytest.mark.parametrize("step", ["sample y", "observe y", "marginalize y"])
def test_primitive_families_reject_static_arguments(step: str) -> None:
    body = f"    {step} <- Normal[Row](0.0, 1.0)\n"
    if step.startswith("marginalize"):
        body += "        observe z <- Normal(0.0, 1.0)\n"
    source = HEAD + "program main : Row -> Out\n" + body + "    return y\nexport main\n"
    with pytest.raises(QiecDiagnosticError, match="accepts no static arguments"):
        _module(source)


def test_typed_array_parameter_is_substituted_inside_draw_indexing() -> None:
    source = (
        HEAD
        + """program sub(x : Tensor[Real]([Row])) : Row -> Out
    let i = 1
    sample z <- Normal(loc=x[i], scale=1.0)
    return z
program main(data : Tensor[Real]([Row])) : Row -> Out
    sample y <- sub(data)
    return y
export main
"""
    )
    run = run_program(
        _module(source), "main", data={"data": (1.0, 2.0, 3.0)}, sites={"y": 2.0}
    )
    assert run.log_joint == pytest.approx(-0.5 * math.log(2.0 * math.pi))


def test_nested_calls_forward_static_object_and_runtime_arguments() -> None:
    source = (
        HEAD
        + TENSOR_SUB
        + """program outer[B : FinSet](input : Tensor[Real]([B])) : Row -> Out
    sample z <- sub[B](input, 0.5)
    return z
program main(data : Tensor[Real]([Row])) : Row -> Out
    sample y <- outer[Row](data)
    return y
export main
"""
    )
    run = run_program(
        _module(source),
        "main",
        data={"data": (1.0, 2.0, 3.0)},
        sites={"y": (1.0, 2.0, 3.0)},
    )
    assert run.value == (1.0, 2.0, 3.0)


def test_static_binders_keep_authored_interleaving() -> None:
    module = parse(
        HEAD
        + """program sub[A : FinSet, n : Nat, B : FinSet](x : Real) : Row -> Out
    return x
"""
    )
    template = next(item for item in module.statements if isinstance(item, ProgramDecl))
    assert [item.name for item in static_program_parameters(template)] == [
        "A",
        "n",
        "B",
    ]
    bindings = template_bindings(
        template,
        (DrawArgScalar(value=2.0),),
        {},
        static_arguments=(
            DrawArgName(text="Row"),
            DrawArgScalar(value=3.0),
            DrawArgName(text="Row"),
        ),
    )
    assert bindings.objects == {"A": "Row", "B": "Row"}
    assert set(bindings.statics) == {"n"}


def test_template_renaming_captures_the_runtime_actual_name() -> None:
    module = parse(
        HEAD
        + """program sub(x : Tensor[Real]([Row])) : Row -> Out
    sample data <- Normal(loc=x[idx], scale=1.0)
    return data
"""
    )
    template = next(item for item in module.statements if isinstance(item, ProgramDecl))
    bindings = template_bindings(
        template,
        (DrawArgName(text="input"),),
        {"input": LetExprVar(name="argument$input")},
    )
    steps = instantiate_program(template, ("output",), bindings)
    assert isinstance(steps[0], SampleStep)
    assert steps[0].args is not None
    argument = steps[0].args[0]
    assert isinstance(argument, DrawArgNamed)
    assert isinstance(argument.value, DrawArgIndex)
    assert argument.value.name == "argument$input"


def test_fractional_static_natural_is_refused_before_substitution() -> None:
    module = parse(HEAD + "program sub[n : Nat](x : Real) : Row -> Out\n    return x\n")
    template = next(item for item in module.statements if isinstance(item, ProgramDecl))
    with pytest.raises(TemplateError, match="requires a natural number"):
        template_bindings(
            template,
            (DrawArgScalar(value=1.0),),
            {},
            static_arguments=(DrawArgScalar(value=1.5),),
        )


NAT_SOURCE = (
    HEAD
    + """program main(n : Nat) : Row -> Out
    return n
export main
"""
)


def test_nat_constraint_survives_canonical_serialization() -> None:
    module = _module(NAT_SOURCE)
    restored = loads(dumps(module))
    assert restored == module
    assert isinstance(restored, QiecModule)
    assert program_entry(restored, "main").parameters[0].constraint == "nonnegative"


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_runtime_nat_rejects_invalid_host_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Model(NAT_SOURCE).bind({"n": value})
    with pytest.raises(
        (ExecutionFailure, TypeError, ValueError), match="nonnegative|integer|Int"
    ):
        run_program(_module(NAT_SOURCE), "main", data={"n": value}, sites={})


@pytest.mark.parametrize("value", [0, 2])
def test_runtime_nat_accepts_natural_host_values(value: int) -> None:
    bound = Model(NAT_SOURCE).bind({"n": value})
    assert bound.schema.entry.parameters[0].constraint == "nonnegative"
    assert (
        run_program(_module(NAT_SOURCE), "main", data={"n": value}, sites={}).value
        == value
    )


@pytest.mark.parametrize("actual", ["-1", "1.5"])
def test_runtime_nat_refuses_invalid_literal_actuals(actual: str) -> None:
    source = (
        HEAD
        + f"""program sub(n : Nat) : Row -> Out
    return n
program main : Row -> Out
    sample y <- sub({actual})
    return y
export main
"""
    )
    with pytest.raises(QiecDiagnosticError):
        _module(source)


def test_runtime_nat_refuses_a_local_integer_without_a_nonnegative_proof() -> None:
    source = (
        HEAD
        + """program sub(n : Nat) : Row -> Out
    return n
program main : Row -> Out
    sample count <- Poisson(rate=1.0)
    sample y <- sub(count)
    return y
export main
"""
    )
    with pytest.raises(QiecDiagnosticError, match="no nonnegative proof"):
        _module(source)


def test_forwarded_nat_constraint_is_preserved() -> None:
    source = (
        HEAD
        + """program sub(n : Nat) : Row -> Out
    return n
program outer(input : Nat) : Row -> Out
    sample z <- sub(input)
    return z
program main : Row -> Out
    sample y <- outer(count)
    return y
export main
"""
    )
    module = _module(source)
    assert program_entry(module, "main").parameters[0].constraint == "nonnegative"
    assert run_program(module, "main", data={"count": 2}, sites={}).value == 2


def test_native_runtime_tensor_subprogram_matches_the_checked_reference() -> None:
    source = (
        HEAD
        + TENSOR_SUB
        + """program main(x : Tensor[Real]([Row])) : Row -> Out
    sample y <- sub[Row](x, 0.5)
    return y
export main
"""
    )
    x = torch.tensor((1.0, 2.0, 3.0))
    y = torch.tensor((1.1, 2.2, 3.3))
    native = Model(source).bind({"x": x}).trace({"y": y})
    reference = run_program(
        _module(source),
        "main",
        data={"x": tuple(x.tolist())},
        sites={"y": tuple(y.tolist())},
    )
    torch.testing.assert_close(native.output, y)
    assert native.log_joint is not None
    assert float(native.log_joint.sum()) == pytest.approx(reference.log_joint)


def test_native_runtime_direct_return_of_tensor_input() -> None:
    source = (
        HEAD
        + """program identity(x : Tensor[Real]([Row])) : Row -> Out
    return x
program main(features : Tensor[Real]([Row])) : Row -> Out
    sample y <- identity(features)
    return y
export main
"""
    )
    value = torch.tensor((1.0, 2.0, 3.0))
    torch.testing.assert_close(
        Model(source).bind({"features": value}).trace().output, value
    )


def test_static_natural_extent_specializes_an_inlined_sample_plate() -> None:
    source = (
        HEAD
        + """program sub[n : Nat](x : Tensor[Real]([n])) : Row -> Out
    sample z : n <- Normal(loc=x, scale=1.0)
    return z
program main(features : Tensor[Real]([Row])) : Row -> Out
    sample y <- sub[3](features)
    return y
export main
"""
    )
    module = _module(source)
    run = run_program(
        module, "main", data={"features": (1.0, 2.0, 3.0)}, sites={"y": (1.0, 2.0, 3.0)}
    )
    assert run.value == (1.0, 2.0, 3.0)
    assert program_entry(module, "main").telescope == ()
    value = torch.tensor((1.0, 2.0, 3.0))
    torch.testing.assert_close(
        Model(source).bind({"features": value}).trace({"y": value}).output, value
    )


@pytest.mark.parametrize(
    ("subprogram", "application", "expected"),
    [
        (
            "program sub[A : Type](x : A) : Row -> Out\n    return x\n",
            "sub[Real](2.0)",
            2.0,
        ),
        ("program sub(flag : Bool) : Row -> Out\n    return flag\n", "sub(true)", True),
        (
            "program sub[A : FinSet](x : Tensor[Real]([A])) : Row -> Out\n    let doubled = x + x\n    return doubled\n",
            "sub[Row]([1.0, 2.0, 3.0])",
            (2.0, 4.0, 6.0),
        ),
    ],
)
def test_runtime_arguments_use_declared_value_types(
    subprogram: str, application: str, expected: object
) -> None:
    source = (
        HEAD
        + subprogram
        + f"program main : Row -> Out\n    sample y <- {application}\n    return y\nexport main\n"
    )
    assert run_program(_module(source), "main", data={}, sites={}).value == expected


@pytest.mark.parametrize("kind", ["FinSet", "Nat"])
def test_natural_extent_can_fill_a_forwarded_static_parameter(kind: str) -> None:
    source = (
        HEAD
        + """program sub[A : FinSet](x : Tensor[Real]([A])) : Row -> Out
    sample z : A <- Normal(loc=x, scale=1.0)
    return z
program outer[n : Nat](x : Tensor[Real]([n])) : Row -> Out
    sample z <- sub[n](x)
    return z
program main(features : Tensor[Real]([Row])) : Row -> Out
    sample y <- outer[3](features)
    return y
export main
"""
    )
    source = source.replace("sub[A : FinSet]", f"sub[A : {kind}]")
    module = _module(source)
    assert run_program(
        module, "main", data={"features": (1.0, 2.0, 3.0)}, sites={"y": (1.0, 2.0, 3.0)}
    ).value == (1.0, 2.0, 3.0)
    value = torch.tensor((1.0, 2.0, 3.0))
    torch.testing.assert_close(
        Model(source).bind({"features": value}).trace({"y": value}).output, value
    )
