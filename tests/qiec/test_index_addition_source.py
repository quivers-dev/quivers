"""Source and host binding use the same checked Nat-addition relations."""

import math

import pytest
import torch

from quivers.dsl import Compiler, CompileError, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.modeling import Model
from quivers.qiec import (
    IndexSum,
    ExecutionFailure,
    dumps,
    loads,
    parse_static_arguments,
    run_named,
    run_program,
    tensor_shape,
)

IDENTITY = """define padded[n : Nat](x : Tensor[Real]([nat_add(n, 1)])) : Tensor[Real]([nat_add(1, n)]) !{} =
    return x
"""
PROGRAM = """object Out : Real 1
object Row : FinSet 2
program padded[n : Nat](x : Tensor[Real]([nat_add(n, 1)])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    return b
export padded
"""


def test_source_fixed_point_core_serialization_and_closed_runtime_static_sum():
    syntax = parse(IDENTITY)
    emitted = module_to_source(syntax)
    assert module_to_source(parse(emitted)) == emitted
    compiler = Compiler(syntax)
    compiler.compile()
    module = compiler.qiec_module
    assert loads(dumps(module)) == module
    assert isinstance(
        tensor_shape(module.computations[0].parameters[0].type)[1][0], IndexSum
    )
    for static in ("n=2", "n=nat_add(1, 1)"):
        arguments = parse_static_arguments(module, "padded", (static,))
        assert run_named(
            module, "padded", ((1.0, 2.0, 3.0),), static_arguments=arguments
        ).value == (1.0, 2.0, 3.0)
    with pytest.raises(ExecutionFailure, match="type|shape|Tensor|length"):
        run_named(module, "padded", ((1.0, 2.0),), static_arguments=arguments)


def test_scoped_sum_extent_is_inferred_consistently_by_native_and_reference_binding():
    model = Model(PROGRAM)
    for width in (1, 4, 7):
        x = torch.ones(width)
        bound = model.bind({"x": x})
        assert bound.schema.describe()["x"]["type"] == f"Tensor[Real]([{width}])"
        module = Compiler(parse(PROGRAM)).qiec_module
        result = run_program(
            module, "padded", data={"x": tuple(x.tolist())}, sites={"b": 0.5}
        )
        assert result.log_joint == pytest.approx(
            -0.5 * (0.5**2 + math.log(2 * math.pi))
        )
    zero = model.bind({"x": torch.ones(1)}, parameters={"n": 0})
    assert zero.schema.entry.telescope == ()
    assert zero.schema.describe()["x"]["type"] == "Tensor[Real]([1])"
    categorical = PROGRAM.replace(": Row -> Out", ": n -> Out")
    with pytest.raises(CompileError, match="FinSet cardinality"):
        Model(categorical).bind({"x": torch.ones(1)}, parameters={"n": 0})
    plain_shape = PROGRAM.replace("nat_add(n, 1)", "n")
    empty = Model(plain_shape).bind({"x": torch.ones(0)}, parameters={"n": 0})
    assert empty.schema.describe()["x"]["type"] == "Tensor[Real]([0])"
    with pytest.raises(
        (CompileError, ValueError), match="nonnegative|solution|inconsistent"
    ):
        model.bind({"x": torch.ones(0)})
    with pytest.raises((CompileError, ValueError), match="solution|inconsistent"):
        model.bind({"x": torch.ones(4)}, parameters={"n": 4})


def test_two_equations_infer_distinct_scoped_extents_and_ambiguous_sum_stays_unbound():
    source = """object Out : Real 1
program sums[n : Nat, m : Nat](
    x : Tensor[Real]([nat_add(n, m)]),
    z : Tensor[Real]([nat_add(n, nat_add(m, m))]),
) : n -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    return b
export sums
"""
    model = Model(source)
    bound = model.bind({"x": torch.ones(5), "z": torch.ones(8)})
    assert bound.schema.entry.telescope == ()
    assert bound.schema.describe()["x"]["type"] == "Tensor[Real]([5])"
    assert bound.schema.describe()["z"]["type"] == "Tensor[Real]([8])"
    # One shape alone cannot identify two unrelated extents.
    ambiguous = source.replace(
        "    z : Tensor[Real]([nat_add(n, nat_add(m, m))]),\n", ""
    )
    with pytest.raises(
        (CompileError, ValueError), match="extent|specializ|static|infer"
    ):
        Model(ambiguous).bind({"x": torch.ones(5)})


@pytest.mark.parametrize(
    "header",
    [
        "[n : Nat](x : Tensor[Real]([nat_add(n, 1, 2)]))",
        "[n : Nat](x : Tensor[Real]([nat_add(m, 1)]))",
        "[s : Shape](x : Tensor[Real]([nat_add(s, 1)]))",
    ],
)
def test_bad_arity_scope_and_sort_fail_at_source_checking(header):
    with pytest.raises(QiecDiagnosticError):
        Compiler(
            parse(f"define bad{header} : Real !{{}} =\n    return 0.0\n")
        ).compile()


def test_user_sort_constructor_named_nat_add_keeps_its_nominal_meaning():
    source = """index Tree = Leaf | nat_add(Tree, Tree)
family Tagged(t : Tree) : Type
    constructor Token : Tagged(nat_add(Leaf, Leaf))
define tag() : Tagged(nat_add(Leaf, Leaf)) !{} =
    return construct Token() as Tagged(nat_add(Leaf, Leaf))
"""
    compiler = Compiler(parse(source))
    compiler.compile()
    module = compiler.qiec_module
    assert loads(dumps(module)) == module
    assert run_named(module, "tag").value is not None


def test_global_inferred_finite_object_can_be_solved_from_an_affine_input_shape():
    source = """object Width : FinSet _
object Row : FinSet 2
object Out : Real 1
program p(x : Tensor[Real]([nat_add(Width, 1)])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    return b
export p
"""
    bound = Model(source).bind({"x": torch.ones(4)})
    assert bound.schema.describe()["x"]["type"] == "Tensor[Real]([4])"
    assert bound.schema.entry.telescope == ()


def test_shape_only_zero_specialization_executes_on_native_and_reference():
    source = PROGRAM.replace("nat_add(n, 1)", "n")
    bound = Model(source).bind({"x": torch.ones(0)}, parameters={"n": 0})
    from quivers.inference.trace import trace

    result = trace(
        bound.program,
        bound.input,
        {"x": torch.ones(0), "b": torch.tensor(0.5)},
    )
    assert result.log_joint.sum().item() == pytest.approx(
        -0.5 * (0.25 + math.log(2 * math.pi))
    )
    module = Compiler(parse(source)).qiec_module
    reference = run_program(module, "padded", data={"x": ()}, sites={"b": 0.5})
    assert reference.log_joint == pytest.approx(-0.5 * (0.25 + math.log(2 * math.pi)))
