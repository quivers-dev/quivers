"""Standalone specializations retain source addresses and nested-call hygiene."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.compiler import Compiler
from quivers.inference import trace
from quivers.modeling import Model
from quivers.qiec.program_runtime import run_program


@pytest.mark.parametrize("kind", ["Nat", "FinSet", "Type"])
@pytest.mark.parametrize("declared_observation", [False, True])
@pytest.mark.parametrize("returned", ["s", "z"])
def test_factory_preserves_observation_names_and_checked_reference(
    kind, declared_observation, returned, monkeypatch
):
    axis = "n" if kind == "Nat" else "Row"
    static = "n : Nat" if kind == "Nat" else f"A : {kind}"
    tensor_axis = "A" if kind == "FinSet" else axis
    header = f"s : Real, x : Tensor[Real]([{tensor_axis}])"
    if kind == "Type":
        header += ", extra : A"
    if declared_observation:
        header += f", y : Tensor[Real]([{tensor_axis}])"
    source = f"""object Row : FinSet 3
object Out : Real 1
program response[{static}]({header}) : {axis} -> Out
    sample z <- Normal(loc=0.0, scale=1.0)
    let mu = x + z
    observe y : {tensor_axis} <- Normal(loc=mu, scale=s)
    return {returned}
export response
"""
    checked = []
    original = Compiler._attach_program_contract

    def capture(self, target, module, entry):
        checked.append((module, entry))
        return original(self, target, module, entry)

    monkeypatch.setattr(Compiler, "_attach_program_contract", capture)
    argument = (
        {"n": 3} if kind == "Nat" else {"A": "Row" if kind == "FinSet" else "Real"}
    )
    program = loads(source).response(**argument).morphism
    values = {
        "s": torch.tensor(0.7, requires_grad=True),
        "x": torch.tensor([0.2, -0.4, 0.8], requires_grad=True),
        "y": torch.tensor([0.1, 0.3, -0.2]),
        "z": torch.tensor([0.25], requires_grad=True),
    }
    if kind == "Type":
        values["extra"] = torch.tensor(0.0)
    result = trace(program, torch.zeros(1, 1), values)
    assert set(result.sites) == {"z", "mu", "y"}
    assert result.sites["y"].is_observed
    assert [
        (p.name, p.role)
        for p in program.input_contract.parameters
        if p.role == "observation"
    ] == [("y", "observation")]
    assert {s.name for s in program.input_contract.sites} == {"z", "y"}
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(values["z"]).sum()
        + torch.distributions.Normal(values["x"] + values["z"], values["s"])
        .log_prob(values["y"])
        .sum()
    )
    torch.testing.assert_close(result.log_joint.sum(), expected)
    differentiated = (values["s"], values["x"], values["z"])
    actual_gradients = torch.autograd.grad(result.log_joint.sum(), differentiated)
    expected_gradients = torch.autograd.grad(expected, differentiated)
    for actual, wanted in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, wanted)
    module, entry = checked[-1]
    reference = run_program(
        module,
        entry.name,
        data={
            "s": 0.7,
            "x": (0.2, -0.4, 0.8),
            "y": (0.1, 0.3, -0.2),
            **({"extra": 0.0} if kind == "Type" else {}),
        },
        sites={"z": 0.25},
    )
    assert reference.log_joint == pytest.approx(expected.item(), abs=1e-6)


@pytest.mark.parametrize("method", ["partial", "full", "define"])
def test_scalar_factory_and_define_application_use_source_observation_names(method):
    source = """object Out : Real 1
program response[n : Nat](s : Real) : n -> Out
    observe y : n <- Normal(loc=0.0, scale=s)
    return s
"""
    if method == "define":
        program = loads(
            source + "define fitted = response[3](sigma)\nexport fitted\n"
        ).morphism
    else:
        factory = loads(source + "export response\n").response
        program = factory(
            **({"n": 3} if method == "partial" else {"n": 3, "s": 0.7})
        ).morphism
    y = torch.tensor([0.1, -0.2, 0.3])
    result = trace(
        program,
        torch.zeros(1, 1),
        {"y": y, "s": torch.tensor(0.7), "sigma": torch.tensor(0.7)},
    )
    assert {site.name for site in program.input_contract.sites} == {"y"}
    assert result.sites["y"].is_observed
    torch.testing.assert_close(
        result.log_joint.sum(), torch.distributions.Normal(0.0, 0.7).log_prob(y).sum()
    )


def test_standalone_outer_factory_preserves_distinct_nested_observations():
    source = """object Out : Real 1
program component[n : Nat](s : Real) : n -> Out
    sample z <- Normal(loc=0.0, scale=1.0)
    observe y : n <- Normal(loc=z, scale=s)
    return z
program outer[n : Nat](s : Real) : n -> Out
    sample a <- component[n](s)
    sample b <- component[n](s)
    return (a, b)
export outer
"""
    program = loads(source).outer(n=3).morphism
    values = {
        "s": torch.tensor(0.7),
        "a": torch.tensor([0.2]),
        "b": torch.tensor([-0.4]),
        "a$y": torch.tensor([0.1, 0.3, -0.2]),
        "b$y": torch.tensor([-0.5, 0.1, 0.8]),
    }
    result = trace(program, torch.zeros(1, 1), values)
    assert set(result.sites) == {"a", "b", "a$y", "b$y"}
    assert all(result.sites[name].is_observed for name in ("a$y", "b$y"))
    assert {
        p.name for p in program.input_contract.parameters if p.role == "observation"
    } == {"a$y", "b$y"}
    expected = sum(
        torch.distributions.Normal(0.0, 1.0).log_prob(values[name]).sum()
        + torch.distributions.Normal(values[name], values["s"])
        .log_prob(values[f"{name}$y"])
        .sum()
        for name in ("a", "b")
    )
    torch.testing.assert_close(result.log_joint.sum(), expected)


def test_standalone_specialization_does_not_capture_a_renamed_actual():
    source = """object Out : Real 1
program response[n : Nat](loc : Real) : n -> Out
    sample z <- Normal(loc=0.0, scale=1.0)
    let mu = loc + z
    observe y : n <- Normal(loc=mu, scale=1.0)
    return loc
define fitted = response[3](z)
export fitted
"""
    program = loads(source).morphism
    values = {
        "z": torch.tensor(0.2),
        "loc$z": torch.tensor([-0.4]),
        "y": torch.tensor([0.1, 0.3, -0.2]),
    }
    result = trace(program, torch.zeros(1, 1), values)
    assert result.sites["y"].is_observed
    assert "loc$z" in result.sites
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(values["loc$z"]).sum()
        + torch.distributions.Normal(values["z"] + values["loc$z"], 1.0)
        .log_prob(values["y"])
        .sum()
    )
    torch.testing.assert_close(result.log_joint.sum(), expected)


def test_legacy_scalar_factory_keeps_its_real_nested_call_namespace():
    source = """object Row : FinSet 3
object Out : Real 1
program component(s : Real) : Row -> Out
    observe y : Row <- Normal(loc=0.0, scale=s)
    return s
program outer(s : Real) : Row -> Out
    sample a <- component(s)
    return a
export outer
"""
    program = loads(source).outer(s=0.7).morphism
    values = {"a$y": torch.tensor([0.1, 0.3, -0.2])}
    result = trace(program, torch.zeros(1, 1), values)
    assert result.sites["a$y"].is_observed
    assert {
        p.name for p in program.input_contract.parameters if p.role == "observation"
    } == {"a$y"}
    torch.testing.assert_close(
        result.log_joint.sum(),
        torch.distributions.Normal(0.0, 0.7).log_prob(values["a$y"]).sum(),
    )


@pytest.mark.parametrize("kind", ["Nat", "FinSet"])
@pytest.mark.parametrize("explicit", [False, True])
def test_model_binding_keeps_inferred_observation_addresses(kind, explicit):
    source = f"""object Out : Real 1
program response[n : {kind}](s : Real) : n -> Out
    observe y : n <- Normal(loc=0.0, scale=s)
    return s
export response
"""
    values = {"s": torch.tensor(0.7), "y": torch.tensor([0.1, -0.2, 0.3])}
    bound = Model(source).bind(values, parameters={"n": 3} if explicit else None)
    result = bound.trace()
    assert bound.schema.observations == ("y",)
    assert result.sites["y"].is_observed
    torch.testing.assert_close(
        result.log_joint.sum(),
        torch.distributions.Normal(0.0, values["s"]).log_prob(values["y"]).sum(),
    )


@pytest.mark.parametrize(
    ("type_name", "extra"),
    [
        ("Real", torch.tensor(0.5)),
        ("Int", torch.tensor(2)),
        ("Bool", torch.tensor(True)),
    ],
)
def test_model_binding_supplies_explicit_static_type_arguments(type_name, extra):
    source = """object Row : FinSet 3
object Out : Real 1
program response[A : Type](extra : A, s : Real) : Row -> Out
    observe y : Row <- Normal(loc=0.0, scale=s)
    return s
export response
"""
    values = {
        "extra": extra,
        "s": torch.tensor(0.7),
        "y": torch.tensor([0.1, -0.2, 0.3]),
    }
    bound = Model(source).bind(values, parameters={"A": type_name})
    assert bound.schema.describe()["extra"]["type"] == type_name
    assert bound.schema.entry.telescope == ()
    assert bound.schema.observations == ("y",)
    result = bound.trace()
    assert result.sites["y"].is_observed
    torch.testing.assert_close(
        result.log_joint.sum(),
        torch.distributions.Normal(0.0, values["s"]).log_prob(values["y"]).sum(),
    )
    rebound = bound.rebind(values)
    assert rebound.schema == bound.schema
    torch.testing.assert_close(rebound.trace().log_joint, result.log_joint)


@pytest.mark.parametrize(("kind", "actual"), [("Type", "Real"), ("Effect", "Random")])
def test_explicit_static_arguments_compose_with_inferred_natural_extents(kind, actual):
    source = f"""object Out : Real 1
program response[A : {kind}, n : Nat](s : Real) : n -> Out
    observe y : n <- Normal(loc=0.0, scale=s)
    return s
export response
"""
    values = {"s": torch.tensor(0.7), "y": torch.tensor([0.1, -0.2, 0.3])}
    bound = Model(source).bind(values, parameters={"A": actual})
    assert bound.schema.entry.telescope == ()
    assert bound.schema.observations == ("y",)
    torch.testing.assert_close(
        bound.trace().log_joint.sum(),
        torch.distributions.Normal(0.0, values["s"]).log_prob(values["y"]).sum(),
    )
