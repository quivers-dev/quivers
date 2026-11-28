"""Native exact enumeration of authored decoder laws and encoder scores."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.qiec import (
    Enumeration,
    ExecutionFailure,
    ExecutionResult,
    FiniteSampleable,
    QiecModule,
    RuntimeSelection,
    dumps,
    loads,
    run_named,
    structural_runtime_configuration,
)
from quivers.qiec.program_runtime import run_program

from tests.dsl.test_authored_decoder_law import _ENCODE_SOURCE


SOURCE = (
    _ENCODE_SOURCE
    + """
handler integrate for Random : LogWeight -> LogWeight [coverage=total, implementation=foreign]
    sample resumes omega

define score_one(code : Tensor[Real]([2]), y : Real) : LogWeight !{random} =
    let point <- draw(code)
    let encoded <- E__encode_value(point)
    return log_prob(Normal(encoded[0], 1.0), y)

define score_two(code : Tensor[Real]([2]), y : Real) : LogWeight !{random} =
    let first <- draw(code)
    let second <- draw(code)
    let first_code <- E__encode_value(first)
    let second_code <- E__encode_value(second)
    return log_prob(Normal(first_code[0] + second_code[0], 1.0), y)

define integrate_one(code : Tensor[Real]([2]), y : Real) : LogWeight !{} =
    handle random with integrate in
        score_one(code, y)

define integrate_two(code : Tensor[Real]([2]), y : Real) : LogWeight !{} =
    handle random with integrate in
        score_two(code, y)

define score_external(law : Sampleable[Tiny__TermValue], y : Real) : LogWeight !{random} =
    let point <- perform random.sample[Tiny__TermValue](site("given"), law)
    let encoded <- E__encode_value(point)
    return log_prob(Normal(encoded[0], 1.0), y)

define integrate_external(law : Sampleable[Tiny__TermValue], y : Real) : LogWeight !{} =
    handle random with integrate in
        score_external(law, y)
"""
)


class _CustomLaw:
    """A sampleable with the correct interface and no ownership assertion."""

    def __init__(self, law: FiniteSampleable) -> None:
        self.law = law
        self.element = law.element
        self.requests = 0

    def sample(self) -> object:
        self.requests += 1
        return self.law.sample()

    def log_prob(self, value: object, keep_batch: bool = False) -> torch.Tensor:
        self.requests += 1
        return self.law.log_prob(value, keep_batch=keep_batch)

    def weighted_support(self) -> tuple[tuple[object, torch.Tensor], ...]:
        self.requests += 1
        return self.law.weighted_support()


@pytest.fixture
def native_decoder():
    compiled = load_source(SOURCE)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    module = module.with_(
        handlers=tuple(
            handler.with_(foreign=Enumeration(reduction="log_expectation"))
            if handler.name == "integrate"
            else handler
            for handler in module.handlers
        )
    )
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    runtime = structural_runtime_configuration(restored, compiled).with_(
        selections=(
            RuntimeSelection(
                provider="core",
                options={"handlers": {"integrate": {"kind": "declared"}}},
            ),
        )
    )
    decoder = compiled.decoders["D"].double()
    encoder = compiled.encoders["E"].double()
    head = decoder._dec_0
    assert isinstance(head, torch.nn.Linear)
    parameters = dict(encoder.named_parameters())
    red, blue = parameters["_op_0.const_Red"], parameters["_op_1.const_Blue"]
    with torch.no_grad():
        head.weight.copy_(
            head.weight.new_tensor([[1.0, -0.5], [-0.25, 0.75], [10.0, 10.0]])
        )
        head.bias.zero_()
        red.copy_(red.new_tensor([-0.5, 1.0]))
        blue.copy_(blue.new_tensor([2.75, -2.0]))
    return restored, runtime, head, red, blue


def test_native_decoder_enumeration_matches_replay_and_exact_joint_oracles(
    native_decoder,
) -> None:
    restored, runtime, head, red, blue = native_decoder
    backend = TensorBackend()
    code = head.weight.new_tensor([0.4, -0.6], requires_grad=True)
    y = code.new_tensor(1.2, requires_grad=True)

    law = run_named(restored, "D__law", (code,), runtime=runtime).value
    support = tuple(point for point, _ in law.weighted_support())
    replay = torch.stack(
        [
            run_program(
                restored,
                "measure",
                data={"code": code, "y": y},
                sites={"decoded": point},
                providers=runtime.providers,
                distributions=backend,
            ).log_joint
            for point in support
        ]
    ).logsumexp(0)
    single = run_named(
        restored, "integrate_one", (code, y), runtime=runtime, distributions=backend
    )
    double = run_named(
        restored, "integrate_two", (code, y), runtime=runtime, distributions=backend
    )

    def score(mean: torch.Tensor) -> torch.Tensor:
        return -0.5 * ((y - mean).square() + math.log(2 * math.pi))

    logits = (head.weight[:2] @ code + head.bias[:2]).log_softmax(0)
    means = torch.stack((red[0], blue[0]))
    expected_one = (logits + score(means)).logsumexp(0)
    expected_two = (
        logits[:, None] + logits[None, :] + score(means[:, None] + means[None, :])
    ).logsumexp((0, 1))
    torch.testing.assert_close(replay, expected_one)
    values = (code, y, head.weight, head.bias, red, blue)
    for actual, expected in (
        (single.value, expected_one),
        (double.value, expected_two),
    ):
        torch.testing.assert_close(actual, expected)
        gradients = torch.autograd.grad(actual, values, retain_graph=True)
        references = torch.autograd.grad(expected, values, retain_graph=True)
        for gradient, reference in zip(gradients, references, strict=True):
            torch.testing.assert_close(gradient, reference)

    def call_count(result: ExecutionResult, name: str) -> int:
        return sum(
            event.event == "call.entered" and event.detail["computation"] == name
            for event in result.trace
        )

    assert call_count(single, "D__law") == 1
    assert call_count(single, "E__encode_value") == 2
    assert call_count(double, "D__law") == 3
    assert call_count(double, "E__encode_value") == 8

    custom = _CustomLaw(law)
    cloned = law.with_(log_weights=law.log_weights.clone())
    for argument in (law, cloned, custom):
        with pytest.raises(
            ExecutionFailure, match="captures nonduplicable local 'law'"
        ):
            run_named(
                restored,
                "integrate_external",
                (argument, y),
                runtime=runtime,
                distributions=backend,
            )
    assert custom.requests == 0


def test_repeated_native_decoder_fitting_uses_current_decoder_and_encoder_weights(
    native_decoder,
) -> None:
    module, runtime, head, red, blue = native_decoder
    backend = TensorBackend()
    code = head.weight.new_tensor([0.4, -0.6])
    observation = code.new_tensor(1.2)
    parameters = (head.weight, head.bias, red, blue)
    reference = tuple(p.detach().clone().requires_grad_() for p in parameters)
    initial = reference
    step_size = 0.15
    optimizer = torch.optim.SGD(parameters, lr=step_size)

    def reference_loss(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        weight, bias, red_code, blue_code = values
        logits = weight[:2] @ code + bias[:2]
        means = torch.stack((red_code[0], blue_code[0]))
        scores = -0.5 * ((observation - means).square() + math.log(2 * math.pi))
        return logits.logsumexp(0) - (logits + scores).logsumexp(0)

    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        result = run_named(
            module,
            "integrate_one",
            (code, observation),
            runtime=runtime,
            distributions=backend,
        )
        expected = reference_loss(reference)
        loss = -result.value
        torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
        loss.backward()
        wanted = torch.autograd.grad(expected, reference)
        for parameter, gradient in zip(parameters, wanted, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, atol=1e-12, rtol=1e-12)
        optimizer.step()
        reference = tuple(
            (value - step_size * gradient).detach().requires_grad_()
            for value, gradient in zip(reference, wanted, strict=True)
        )
        for parameter, value in zip(parameters, reference, strict=True):
            torch.testing.assert_close(parameter, value, atol=1e-12, rtol=1e-12)
        losses.append(loss.detach())
    assert float(losses[-1]) < float(losses[0])
    fitted = reference_loss(reference)
    assert not torch.isclose(fitted, reference_loss((*initial[:2], *reference[2:])))
    assert not torch.isclose(fitted, reference_loss((*reference[:2], *initial[2:])))
