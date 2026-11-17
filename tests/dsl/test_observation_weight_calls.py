"""Authored pure calls preserve checked observation-weight operations."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.dsl import loads, parse
from quivers.dsl.pure_builtins import EAGER_BUILTINS
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.program_runtime import run_program
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

SOURCE = """define reweight(w : Real, density : Real) : Real !{} =
    return weight_value(scale_weight(observation_weight(w), weight(density)))
object Row : FinSet 3
object Out : Real 1
program main(w : Real, density : Real) : Row -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
    let contribution <- reweight(w, density)
    score weighted = contribution
    return a
export main
"""


@pytest.mark.parametrize(
    ("weight", "density"), [(0.0, -math.inf), (0.5, -3.0), (2.0, -3.0)]
)
def test_authored_weight_operations_execute_natively_and_on_reference(
    weight, density
) -> None:
    model = loads(SOURCE)
    program = model.morphism
    assert isinstance(program, MonadicProgram)
    expected = -0.5 * math.log(2.0 * math.pi) + (
        0.0 if weight == 0 else weight * density
    )
    actual = program.log_joint(
        torch.zeros(1, 1),
        {
            "a": torch.tensor([0.0]),
            "w": torch.tensor(weight),
            "density": torch.tensor(density),
        },
    )
    module = lower_qvr_to_qiec(parse(SOURCE))
    reference = run_program(
        module, "main", data={"w": weight, "density": density}, sites={"a": 0.0}
    )
    assert float(actual.sum()) == pytest.approx(expected)
    assert reference.log_joint == pytest.approx(expected)


@pytest.mark.parametrize("weight", [-1.0, math.nan, math.inf])
def test_authored_weight_operations_validate_named_values_on_both_executors(
    weight,
) -> None:
    program = loads(SOURCE).morphism
    with pytest.raises(ValueError, match="finite and nonnegative"):
        program.log_joint(
            torch.zeros(1, 1),
            {
                "a": torch.tensor([0.0]),
                "w": torch.tensor(weight),
                "density": torch.tensor(-3.0),
            },
        )
    module = lower_qvr_to_qiec(parse(SOURCE))
    with pytest.raises(ExecutionFailure, match="finite and nonnegative"):
        run_program(
            module, "main", data={"w": weight, "density": -3.0}, sites={"a": 0.0}
        )


def test_native_scaling_masks_before_multiplication_and_has_finite_gradients() -> None:
    weights = torch.tensor([0.0, 2.0], requires_grad=True)
    density = torch.tensor([-torch.inf, -3.0], requires_grad=True)
    checked = EAGER_BUILTINS["observation_weight"](weights)
    result = EAGER_BUILTINS["scale_weight"](checked, density)
    assert torch.equal(result, torch.tensor([0.0, -6.0]))
    weight_grad, density_grad = torch.autograd.grad(result.sum(), (weights, density))
    assert torch.equal(weight_grad, torch.tensor([0.0, -3.0]))
    assert torch.equal(density_grad, torch.tensor([0.0, 2.0]))


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([1, 2]),
        torch.tensor([True]),
        torch.tensor([0.0, -1.0]),
        torch.tensor([math.inf]),
    ],
)
def test_native_checked_weight_identity_refuses_invalid_tensors(value) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        EAGER_BUILTINS["observation_weight"](value)


@pytest.mark.parametrize("target", tuple(available_targets()))
@pytest.mark.parametrize("offset", ["", " + 1.0"])
def test_authored_weight_primitives_use_canonical_runtime_in_external_targets(
    target,
    offset,
) -> None:
    source = SOURCE.replace(
        "return weight_value(scale_weight(observation_weight(w), weight(density)))",
        "return weight_value(scale_weight(observation_weight(w), weight(density)))"
        + offset,
    )
    if target in {"stan", "bugs", "jags"}:
        with pytest.raises(UnsupportedConstruct, match="weight|computation|graph"):
            transpile(parse(source), target=target)
        return
    result = transpile(parse(source), target=target).decode()
    assert "observation_weight" in result or "observation-weight" in result
    assert "scale_weight" in result or "scale-weight" in result
    assert "torch.observation_weight" not in result
    assert "torch.scale_weight" not in result
