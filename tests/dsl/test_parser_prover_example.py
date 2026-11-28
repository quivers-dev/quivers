"""The documented parser feeds each reading to a separate Boolean prover."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.qiec.checking import KernelError
from quivers.qiec.execution import ExecutionFailure, run_named
from quivers.qiec.foreign import DeductionOutcomeLaw
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.program_runtime import run_program, sample_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import BOOL

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


def test_weighted_reading_law_draws_and_scores_the_same_nominal_forms() -> None:
    compiled = load_source(
        _example()
        + """
define approved_form() : Reading__Item !{} =
    return construct Reading__approved() as Reading__Item

define unresolved_form() : Reading__Item !{} =
    return construct Reading__unresolved() as Reading__Item
"""
    )
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule)
    forms = tuple(
        run_named(module, name).value for name in ("approved_form", "unresolved_form")
    )
    weights = torch.tensor([-0.3, 0.7, -0.4], dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([0.82, 0.17], dtype=weights.dtype, requires_grad=True)
    parameters = {
        f"Reading.lex.{row}": weight for row, weight in enumerate(weights.unbind())
    }
    for observed in (False, True):
        data = {
            "first": "item",
            "last": "reviewed",
            "response": observed,
            "yes": rates[0],
            "no": rates[1],
        }
        options = dict(data=data, parameters=parameters, distributions=TensorBackend())
        replayed = tuple(
            run_program(module, "assess_drawn", sites={"reading": form}, **options)
            for form in forms
        )
        likelihoods = rates if observed else 1 - rates
        joint = torch.stack(tuple(run.log_joint for run in replayed))
        expected = torch.stack((weights[:2].logsumexp(0), weights[2]))
        expected = expected - weights.logsumexp(0) + likelihoods.log()
        assert replayed[0].value == rates[0] and replayed[1].value == rates[1]
        torch.testing.assert_close(joint, expected)

        draw = sample_program(module, "assess_drawn", **options)
        selected = 0 if draw.value == rates[0] else 1
        assert draw.value == rates[selected]
        torch.testing.assert_close(draw.log_joint, replayed[selected].log_joint)
        actual = joint.logsumexp(0)
        oracle = (weights + likelihoods[[0, 0, 1]].log()).logsumexp(0)
        oracle = oracle - weights.logsumexp(0)
        torch.testing.assert_close(actual, oracle)
        actual_grad = torch.autograd.grad(actual, (weights, rates), retain_graph=True)
        oracle_grad = torch.autograd.grad(oracle, (weights, rates))
        for found, reference in zip(actual_grad, oracle_grad, strict=True):
            torch.testing.assert_close(found, reference)


def test_outcome_law_refuses_empty_rows_and_unrelated_carriers() -> None:
    compiled = load_source(_example())
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    with pytest.raises(ExecutionFailure, match="no positive mass"):
        sample_program(
            module,
            "assess_drawn",
            data={
                "first": "missing",
                "last": "reviewed",
                "response": True,
                "yes": 0.8,
                "no": 0.3,
            },
        )
    wrong = _example().replace("outcome_law(readings)", "outcome_law(yes)")
    with pytest.raises(
        QiecDiagnosticError, match="requires checked weighted deduction outcomes"
    ):
        load_source(wrong)


def test_outcome_law_accepts_rows_grouped_by_a_checked_key() -> None:
    source = (
        _example()
        + """
define meaning(item : Reading__Item) : Bool !{} =
    return true

define draw_meaning(law : Sampleable[Bool]) : Real !{random} =
    let value <- perform random.sample[Bool](site("meaning"), law)
    if value then
        return 1.0
    else
        return 0.0

program assess_grouped(first : String, last : String) : Word -> Weight
    let sentence <- pack_tokens(first, last)
    let readings = parse_outcomes(Reading, sentence, meaning)
    let law = outcome_law(readings)
    let prediction <- draw_meaning(law)
    return prediction

export assess_grouped
"""
    )
    module = loads(dumps(load_source(source).qiec))
    assert isinstance(module, QiecModule)
    weights = torch.tensor([-1.2, 1.0, -0.2], dtype=torch.float64, requires_grad=True)
    data = {"first": "item", "last": "reviewed"}
    parameters = {
        f"Reading.lex.{row}": weight for row, weight in enumerate(weights.unbind())
    }
    result = run_program(
        module,
        "assess_grouped",
        data=data,
        sites={"meaning": True},
        parameters=parameters,
    )
    assert result.value == 1.0 and result.log_joint == 0.0
    (gradient,) = torch.autograd.grad(result.log_joint, weights)
    torch.testing.assert_close(gradient, torch.zeros_like(weights))


def test_serialized_outcome_law_cannot_claim_another_result_carrier() -> None:
    compiled = load_source(_example())
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    (handler,) = (
        item
        for item in module.handlers
        if isinstance(item.foreign, DeductionOutcomeLaw)
    )
    law = handler.foreign
    assert isinstance(law, DeductionOutcomeLaw)
    forged = handler.with_(foreign=law.with_(layout=law.layout.with_(key_type=BOOL)))
    changed = module.with_(
        handlers=tuple(
            forged if item.id == handler.id else item for item in module.handlers
        )
    )
    with pytest.raises(KernelError, match="outcome law has an invalid query signature"):
        validate_module(changed)
