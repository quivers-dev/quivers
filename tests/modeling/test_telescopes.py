"""Program interfaces specialize finite indices without changing their kernels."""

import math

import pytest
import torch

from quivers.dsl import CompileError, ParseError, loads, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.modeling import Model


SOURCE = """object Out : Real 1
program regression[n : Nat](x : Tensor[Real]([n]), y : Tensor[Real]([n])) : n -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : n <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""


@pytest.mark.parametrize("size", [1, 3, 7])
def test_static_natural_extent_has_an_independent_density_and_gradient_oracle(size):
    data = {"x": torch.linspace(0.5, 1.5, size), "y": torch.linspace(-0.5, 0.5, size)}
    specification = Model(SOURCE)
    bound = specification.bind(data)
    assert bound.schema.describe()["x"]["type"] == f"Tensor[Real]([{size}])"
    assert specification.schema.describe()["x"]["type"] == "Tensor[Real]([n])"
    b = torch.tensor([0.25], requires_grad=True)
    density = bound.trace({"b": b}).log_joint.sum()
    expected = -0.5 * (
        b.square().sum()
        + (data["y"] - b * data["x"]).square().sum()
        + (size + 1) * math.log(2 * math.pi)
    )
    torch.testing.assert_close(density, expected)
    torch.testing.assert_close(
        torch.autograd.grad(density, b)[0],
        torch.autograd.grad(expected, b)[0],
    )


def test_static_object_binder_uses_the_same_checked_shape_equations():
    source = SOURCE.replace("n : Nat", "n : FinSet")
    bound = Model(source).bind({"x": torch.ones(3), "y": torch.zeros(3)})
    assert bound.schema.describe()["x"]["type"] == "Tensor[Real]([3])"
    assert bound.schema.entry.telescope == ()


def test_explicit_static_argument_must_agree_with_runtime_shape():
    model = Model(SOURCE)
    data = {"x": torch.ones(3), "y": torch.zeros(3)}
    explicit = model.bind(data, parameters={"n": 3})
    torch.testing.assert_close(
        explicit.trace({"b": torch.zeros(1)}).log_joint,
        torch.tensor([-4 * 0.5 * math.log(2 * math.pi)]),
    )
    with pytest.raises(CompileError, match="cannot infer.*n"):
        model.bind(data, parameters={"n": 4})


@pytest.mark.parametrize("bad", [True, -1, 1.5])
def test_static_extent_is_a_natural_number(bad):
    with pytest.raises(TypeError, match="nonnegative integer"):
        Model(SOURCE).bind(
            {"x": torch.ones(3), "y": torch.zeros(3)}, parameters={"n": bad}
        )


def test_static_binders_are_scoped_per_program():
    source = SOURCE.replace("export regression", "") + SOURCE.replace(
        "object Out : Real 1\n", ""
    ).replace("regression", "other").replace("x : Tensor", "z : Tensor").replace(
        "b * x", "b * z"
    ).replace("y : Tensor", "w : Tensor").replace("observe y", "observe w")
    bound = Model(source, entry="regression").bind(
        {"x": torch.ones(3), "y": torch.zeros(3)}
    )
    assert bound.schema.describe()["x"]["type"] == "Tensor[Real]([3])"


def test_scalar_values_are_runtime_inputs_and_preserve_autograd():
    source = SOURCE.replace("(x : Tensor", "(offset : Real, x : Tensor").replace(
        "b * x", "b * x + offset"
    )
    offset = torch.tensor(0.25, requires_grad=True)
    data = {"x": torch.ones(3), "y": torch.zeros(3), "offset": offset}
    bound = Model(source).bind(data)
    bound.trace({"b": torch.zeros(1)}).log_joint.sum().backward()
    torch.testing.assert_close(offset.grad, torch.tensor(-0.75))
    with pytest.raises(ValueError, match="runtime values in data"):
        Model(source).bind(data, parameters={"offset": 0.25})


def test_unresolved_static_arguments_request_binding_evidence():
    source = """object One : FinSet 1
object Out : Real 1
program model[n : Nat] : One -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    return b
export model
"""
    with pytest.raises(CompileError, match="cannot infer static arguments"):
        Model(source).bind({})
    assert loads(source).model(n=2).morphism is not None


def test_header_roundtrip_and_static_runtime_name_collision():
    canonical = module_to_source(parse(SOURCE))
    assert module_to_source(parse(canonical)) == canonical
    with pytest.raises(QiecDiagnosticError, match="duplicate program parameter"):
        Model(SOURCE.replace("(x : Tensor", "(n : Real, x : Tensor"))
    with pytest.raises(ParseError):
        parse(SOURCE.replace("    sample b", "    data z : Real\n    sample b"))


def test_inferred_entry_preserves_generic_calls_without_capturing_global_names():
    source = (
        "define __qvr_template_regression() : Real !{} =\n    return 0.0\n"
        + SOURCE.replace("export regression", "")
        + """object Row : FinSet 3
program caller(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- regression[3](x, y)
    return b
export caller
"""
    )
    bound = Model(source, entry="caller").bind(
        {"x": torch.ones(3), "y": torch.zeros(3)}
    )
    density = bound.trace({"b": torch.tensor([0.25])}).log_joint
    expected = -0.5 * (4 * 0.25**2 + 4 * math.log(2 * math.pi))
    torch.testing.assert_close(density, torch.tensor([expected]))
