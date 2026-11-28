"""The documented parser feeds each reading to a separate Boolean prover."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads

GUIDE = Path(__file__).resolve().parents[2] / "docs/guides/deduction.md"


def _example() -> str:
    """Read the standalone QVR block in the prover example."""
    section = GUIDE.read_text(encoding="utf-8").split(
        "#### Scoring a response with a prover\n", 1
    )[1]
    return section.split("\n```qvr\n", 1)[1].split("\n```", 1)[0] + "\n"


def test_example_scores_a_response_from_the_proved_readings() -> None:
    compiled = load_source(_example())
    yes, no = 0.8, 0.3
    observed = {"first": "item", "last": "reviewed", "yes": yes, "no": no}
    public = compiled.run("assess", data=observed)
    response = torch.tensor((2 * yes + no) / 3, dtype=torch.float64)
    assert public.log_joint == pytest.approx(response.log().item())
    boundaries = {"pack_tokens", "Reading__outcomes", "to_claim", "Prover__run"}
    calls = [
        event.detail["computation"]
        for event in public.result.trace
        if event.event == "call.entered" and event.detail["computation"] in boundaries
    ]
    assert calls == [
        "pack_tokens",
        "Reading__outcomes",
        "to_claim",
        "Prover__run",
        "to_claim",
        "Prover__run",
    ]

    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert restored == module
    weights = torch.tensor([-0.2, -1.1, -0.6], dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([yes, no], dtype=weights.dtype, requires_grad=True)
    parameters = {
        f"Reading.lex.{row}": weight for row, weight in enumerate(weights.unbind())
    }
    run = run_program(
        restored,
        "assess",
        data={**observed, "yes": rates[0], "no": rates[1]},
        sites={},
        parameters=parameters,
    )
    # Enumerate the three proofs. The first two yield approved, whose
    # claim gives Licensed and Goal. The third gives only Pending.
    proof_responses = rates[[0, 0, 1]].log()
    expected = torch.logsumexp(weights + proof_responses, 0) - torch.logsumexp(
        weights, 0
    )
    torch.testing.assert_close(run.log_joint, expected, atol=1e-12, rtol=1e-12)
    actual_grad = torch.autograd.grad(run.log_joint, (weights, rates))
    expected_grad = torch.autograd.grad(expected, (weights, rates))
    for found, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(found, reference, atol=1e-12, rtol=1e-12)


def test_example_requires_the_explicit_typed_adapter() -> None:
    source = _example().replace(
        "Prover__run[1]([claim], [true])", "Prover__run[1]([form], [true])"
    )
    with pytest.raises(
        QiecDiagnosticError,
        match=r"expected Tensor\[Prover__Item\].*got Tensor\[Reading__Item\]",
    ) as error:
        load_source(source)
    assert error.value.code == "qiec-call"


def test_neural_observation_backpropagates_through_parser_and_prover() -> None:
    compiled = load_source(_example())
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    dtype = torch.float64
    weights = torch.tensor([-0.2, -1.1, -0.6], dtype=dtype, requires_grad=True)
    network = torch.tensor([0.10, 0.55, -0.26, 0.86], dtype=dtype, requires_grad=True)
    observed = {"first": "item", "last": "reviewed", "response": True}

    def parameters():
        return {
            **{
                f"Reading.lex.{row}": weight
                for row, weight in enumerate(weights.unbind())
            },
            "response.network": network,
        }

    initial = run_program(
        module,
        "assess_neural",
        data=observed,
        sites={},
        parameters={name: value.detach() for name, value in parameters().items()},
    )
    wrong_source = _example().replace(
        "neural_response(proved, theta)", "neural_response(true, theta)"
    )
    wrong_module = load_source(wrong_source).qiec
    assert isinstance(wrong_module, QiecModule)
    wrong = run_program(
        wrong_module,
        "assess_neural",
        data=observed,
        sites={},
        parameters=parameters(),
        distributions=TensorBackend(),
    )
    assert not torch.isclose(wrong.log_joint.detach(), initial.log_joint)
    (erased,) = torch.autograd.grad(wrong.log_joint, weights)
    torch.testing.assert_close(erased, torch.zeros_like(erased), rtol=0, atol=1e-14)
    paths = weights.new_tensor([1.0, 1.0, -1.0])
    for iteration in range(3):
        response = iteration != 1
        run = run_program(
            restored,
            "assess_neural",
            data={**observed, "response": response},
            sites={},
            parameters=parameters(),
            distributions=TensorBackend(),
        )
        # The three derivations remain explicit in this oracle. The first
        # two prove the goal and the last fails, giving the three features
        # in proof order. The observed Bernoulli integrates those proofs.
        prediction = torch.sigmoid(
            network[0] + network[1] * torch.tanh(network[2] + network[3] * paths)
        )
        conditional = prediction if response else 1.0 - prediction
        expected = (weights + conditional.log()).logsumexp(0) - weights.logsumexp(0)
        torch.testing.assert_close(run.log_joint, expected)
        if iteration == 0:
            assert initial.log_joint == pytest.approx(float(expected.detach()))
            boundaries = {
                "pack_tokens",
                "Reading__outcomes",
                "read_network",
                "to_claim",
                "Prover__run",
                "neural_response",
            }
            calls = [
                event.detail["computation"]
                for event in run.result.trace
                if event.event == "call.entered"
                and event.detail["computation"] in boundaries
            ]
            assert calls == [
                "pack_tokens",
                "Reading__outcomes",
                "read_network",
                "to_claim",
                "Prover__run",
                "neural_response",
                "to_claim",
                "Prover__run",
                "neural_response",
            ]
        actual_gradient = torch.autograd.grad(run.log_joint, (weights, network))
        reference_gradient = torch.autograd.grad(expected, (weights, network))
        for actual, wanted in zip(actual_gradient, reference_gradient, strict=True):
            assert bool((wanted.abs() > 1e-8).all())
            torch.testing.assert_close(actual, wanted)
        with torch.no_grad():
            weights.add_(0.1 * actual_gradient[0])
            network.add_(0.2 * actual_gradient[1])
