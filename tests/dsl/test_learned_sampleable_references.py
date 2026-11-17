"""Nested constructors consume actual shared learned kernels at original inputs."""

import pytest
import torch

from quivers.dsl import parse
from quivers.dsl.compiler.core import Compiler
from quivers.inference.trace import trace


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("rows", [2, 3])
@pytest.mark.parametrize("source_kind", ["linear", "mlp(3)"])
@pytest.mark.parametrize(
    "operator",
    [
        "Restrict(base=base,low=-1.0,high=1.0)",
        "Normalize(base=Restrict(base=alias,low=lower,high=1.0))",
    ],
)
def test_learned_sampleable_reference_uses_registered_kernel_context_and_all_gradients(
    operator, width, rows, source_kind
):
    source = f"""object Input:Real 2
object Output:Real {width}
morphism base:Input->Output [param_source={source_kind}] ~ Normal
define alias = base
program model(lower:Real):Input->Output
    sample b <- {operator}
    return b
export model
"""
    compiler = Compiler(parse(source))
    program = compiler.compile()
    kernel = compiler.morphisms["base"]
    program.morphism.double()
    x = torch.tensor(
        [[0.3, -0.2], [0.1, 0.7], [0.2, 0.6]], dtype=torch.double, requires_grad=True
    )[:rows]
    lower = torch.tensor(-0.8, dtype=torch.double, requires_grad=True)
    y = torch.tensor([[0.2], [-0.1], [0.15]], dtype=torch.double)[:rows].expand(
        -1, width
    )
    base = kernel.sampling_distribution(x)
    low = lower if "lower" in operator else lower.new_tensor(-1.0)
    oracle = (
        base.log_prob(y) - (base.cdf(y.new_tensor(1.0)) - base.cdf(low)).log()
    ).sum()
    actual = trace(program.morphism, x, {"lower": lower, "b": y}).log_joint.sum()
    torch.testing.assert_close(actual, oracle)
    leaves = (x, *tuple(kernel.parameters())) + (
        (lower,) if "lower" in operator else ()
    )
    actual_grad = torch.autograd.grad(actual, leaves, retain_graph=True)
    oracle_grad = torch.autograd.grad(oracle, leaves)
    for actual_part, expected_part in zip(actual_grad, oracle_grad, strict=True):
        torch.testing.assert_close(actual_part, expected_part)
    linears = (
        [kernel.param_source.linear]
        if source_kind == "linear"
        else [
            module
            for module in kernel.param_source.net
            if isinstance(module, torch.nn.Linear)
        ]
    )
    bindings = {"input": tuple(x[0].detach().tolist()), "lower": lower.item()}
    for index, layer in enumerate(linears):
        prefix = (
            "base_param" if index == len(linears) - 1 else f"base_param_layer{index}"
        )
        bindings[prefix + "_weight"] = tuple(
            tuple(row) for row in layer.weight.detach().tolist()
        )
        bindings[prefix + "_bias"] = tuple(layer.bias.detach().tolist())
    reference = program.run("model", data=bindings, sites={"b": tuple(y[0].tolist())})
    first = (base.log_prob(y) - (base.cdf(y.new_tensor(1.0)) - base.cdf(low)).log())[
        0
    ].sum()
    assert reference.log_joint == pytest.approx(first.item(), abs=1e-10)
    assert any(module is kernel for module in program.morphism.modules())
    assert len(list(program.morphism.parameters())) == len(
        {id(parameter) for parameter in kernel.parameters()}
    )


def test_learned_sampleable_references_share_one_context_without_constructor_probes():
    from unittest.mock import patch

    source = """object Input:Real 2
object Output:Real 1
morphism base:Input->Output [param_source=linear] ~ Normal
define alias = base
program model:Input->Output
    sample b <- Mixture(weights=[0.3,0.7],components=[base,alias])
    return b
export model
"""
    compiler = Compiler(parse(source))
    # Constructor support is taken from metadata, never from evaluating a
    # learned head at a fabricated input or drawing a probing sample.
    from quivers.continuous.morphisms import ContinuousMorphism

    with patch.object(
        ContinuousMorphism,
        "sampling_distribution",
        side_effect=AssertionError("constructor probe"),
    ):
        program = compiler.compile()
    kernel = compiler.morphisms["base"]
    program.morphism.double()
    x = torch.tensor(
        [[0.3, -0.2], [0.1, 0.7], [0.8, -0.4]], dtype=torch.double, requires_grad=True
    )
    y = torch.tensor([[0.2], [-0.1], [0.4]], dtype=torch.double)
    actual = trace(program.morphism, x, {"b": y}).log_joint.sum()
    oracle = kernel.sampling_distribution(x).log_prob(y).sum()
    torch.testing.assert_close(actual, oracle)
    for a, b in zip(
        torch.autograd.grad(actual, (x, *kernel.parameters()), retain_graph=True),
        torch.autograd.grad(oracle, (x, *kernel.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    assert len(list(program.morphism.parameters())) == len(list(kernel.parameters()))
    mixed = next(
        module
        for module in program.morphism.modules()
        if hasattr(module, "_param_spec")
    )
    assert sum(kind == "var" for kind, _ in mixed._param_spec) == 1
    weights = tuple(
        tuple(row) for row in kernel.param_source.linear.weight.detach().tolist()
    )
    biases = tuple(kernel.param_source.linear.bias.detach().tolist())
    reference = program.run(
        "model",
        data={
            "input": (0.3, -0.2),
            "base_param_weight": weights,
            "base_param_bias": biases,
        },
        sites={"b": (0.2,)},
    )
    assert reference.log_joint == pytest.approx(
        kernel.sampling_distribution(x[:1]).log_prob(y[:1]).sum().item(), abs=1e-10
    )
    from quivers.qiec.execution import ExecutionFailure

    with pytest.raises(ExecutionFailure, match="base_param_weight"):
        program.run("model", data={"input": (0.3, -0.2)}, sites={"b": (0.2,)})


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize(
    "constructor", ["base", "Restrict(base=base,low=-1.0,high=1.0)"]
)
def test_model_schema_binding_and_inference_retain_native_parameter_ownership(
    declared, constructor
):
    from quivers.modeling import Model, VariationalMethod

    header = (
        "(base_param_weight:Tensor[Real]([2,2]),base_param_bias:Tensor[Real]([2]))"
        if declared
        else ""
    )
    source = f"""object Input:Real 2
object Output:Real 1
morphism base:Input->Output [param_source=linear] ~ Normal
program model{header}:Input->Output
    sample b <- Normal(loc=0.0,scale=1.0)
    observe y <- {constructor}
    return b
export model
"""
    model = Model(source)
    assert set(model.schema.describe()) == {"y"}
    x = torch.tensor([[0.3, -0.2]])
    y = torch.tensor([0.2])
    bound = model.bind({"y": y}, input=x)
    result = bound.trace({"b": torch.tensor(0.1)})
    assert torch.isfinite(result.log_joint).all()
    fit = bound.infer(VariationalMethod(num_steps=2))
    assert len(fit.losses) == 2 and fit.draws(num_samples=2)["b"].shape[:2] == (1, 2)
    with pytest.raises(
        ValueError,
        match="unknown external|unexpected|unknown data|unrecognized|not declared|unknown input",
    ):
        model.bind({"y": y, "base_param_weight": torch.zeros(2, 2)}, input=x)


def test_finite_input_context_retains_lookup_carrier_and_table_gradients():
    source = """object Person:FinSet 3
object Output:Real 2
morphism base:Person->Output ~ Normal
program model:Person->Output
    sample b <- Restrict(base=base,low=-1.0,high=1.0)
    return b
export model
"""
    compiler = Compiler(parse(source))
    program = compiler.compile()
    kernel = compiler.morphisms["base"]
    program.morphism.double()
    x = torch.tensor([[0], [2]], dtype=torch.long)
    y = torch.tensor([[0.2, -0.3], [0.1, 0.4]], dtype=torch.double)
    distribution = kernel.sampling_distribution(x)
    oracle = (
        distribution.log_prob(y)
        - (
            distribution.cdf(y.new_tensor(1.0)) - distribution.cdf(y.new_tensor(-1.0))
        ).log()
    ).sum()
    actual = trace(program.morphism, x, {"b": y}).log_joint.sum()
    torch.testing.assert_close(actual, oracle)
    for actual_part, expected_part in zip(
        torch.autograd.grad(actual, tuple(kernel.parameters()), retain_graph=True),
        torch.autograd.grad(oracle, tuple(kernel.parameters())),
        strict=True,
    ):
        torch.testing.assert_close(actual_part, expected_part)
