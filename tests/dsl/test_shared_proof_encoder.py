"""One weighted parse supplies a typed meaning and a nonlinear proof code."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.effects.checked_program import CheckedProgram
from quivers.modeling import Model
from quivers.qiec import (
    ProgramRun,
    QiecModule,
    TypeApplication,
    dumps,
    loads,
    run_program,
)
from quivers.qiec.foreign import DeductionProofs
from quivers.qiec.rules import RulePresentation


_SOURCE = (
    Path(__file__).resolve().parents[2] / "docs/examples/qiec/shared-proof-encoder.qvr"
)


def test_one_presentation_drives_meaning_and_derivation_encoding() -> None:
    source = _SOURCE.read_text()
    compiled = load_source(source)
    original = compiled.qiec
    assert isinstance(original, QiecModule)
    module = loads(dumps(original))
    assert isinstance(module, QiecModule) and module == original
    (descriptor,) = (
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionProofs)
    )
    authored = compiled.deductions["Mix"].presentation
    chart = descriptor.chart.presentation
    qualified = descriptor.proof_presentation
    assert isinstance(authored, RulePresentation)
    assert isinstance(chart, RulePresentation)
    assert authored.id == chart.id == qualified.id
    assert (
        {rule.id for rule in authored.rules}
        == {rule.id for rule in chart.rules}
        == {rule.id for rule in qualified.rules}
    )
    (proof_row,) = [
        declaration
        for declaration in module.constructors
        if declaration.name == "Mix__proof_entry"
    ]
    proof_type = proof_row.fields[0].type
    assert isinstance(proof_type, TypeApplication)
    interpreted = [
        declaration
        for declaration in module.computations
        if declaration.name in {"meaning", "embed_proof"}
    ]
    assert len(interpreted) == 2
    for declaration in interpreted:
        input_type = declaration.parameters[0].type
        assert isinstance(input_type, TypeApplication)
        assert input_type.constructor.id == proof_type.constructor.id
    bound = Model(source, entry="assess_encoded").bind(
        {"sentence": ("x", "y"), "response": True}
    )
    assert isinstance(bound.program, CheckedProgram)

    weights = torch.tensor([0.3, -0.1], dtype=torch.float64, requires_grad=True)
    network = torch.tensor(
        [-0.3, 0.9, 0.2, -0.45], dtype=weights.dtype, requires_grad=True
    )
    codes = weights.new_tensor([[7.0, 2.0], [11.0, -2.0]])
    backend = TensorBackend()
    for observed in (True, False):
        result = _score(module, weights, network, observed, backend)
        probability = torch.sigmoid(
            network[0] + network[1] * torch.tanh(codes @ network[2:])
        )
        torch.testing.assert_close(result.value, weights.softmax(0) @ probability)
        conditional = probability if observed else 1.0 - probability
        expected = (weights + conditional.log()).logsumexp(0) - weights.logsumexp(0)
        torch.testing.assert_close(result.log_joint, expected)
        for gradient, reference in zip(
            torch.autograd.grad(result.log_joint, (weights, network)),
            torch.autograd.grad(expected, (weights, network)),
            strict=True,
        ):
            torch.testing.assert_close(gradient, reference)
            assert bool((gradient.abs() > 1e-8).all())
        calls = Counter(
            event.detail["computation"]
            for event in result.result.trace
            if event.event == "call.entered"
        )
        assert calls["Mix__proofs"] == 1
        assert calls["meaning"] == calls["embed_proof"] == 6
        assert calls["approve"] == 2
        assert calls["leaf_meaning"] == calls["encode_word"] == 4
        assert calls["square_left"] == calls["square_right"] == 1

    mean_code = weights.softmax(0) @ codes
    prediction_at_mean = torch.sigmoid(
        network[0] + network[1] * torch.tanh(mean_code @ network[2:])
    )
    assert not torch.isclose(prediction_at_mean, result.value)

    fixed_source = source.replace(
        "let code <- embed_proof[i, j, category, form](proof)",
        "let code = [7.0, 2.0]",
    )
    fixed_module = load_source(fixed_source).qiec
    assert isinstance(fixed_module, QiecModule)
    fixed = _score(fixed_module, weights, network, True, backend)
    assert not torch.isclose(fixed.value, result.value)
    (erased,) = torch.autograd.grad(fixed.log_joint, weights)
    torch.testing.assert_close(erased, torch.zeros_like(weights), atol=1e-14, rtol=0.0)

    changed_meaning = source.replace(
        "return construct approved() as Meaning",
        'return construct word("unlicensed") as Meaning',
    )
    changed_module = load_source(changed_meaning).qiec
    assert isinstance(changed_module, QiecModule)
    changed = _score(changed_module, weights, network, True, backend)
    torch.testing.assert_close(changed.value, weights.new_tensor(0.5))


def _score(
    module: QiecModule,
    weights: torch.Tensor,
    network: torch.Tensor,
    observed: bool,
    backend: TensorBackend,
) -> ProgramRun:
    return run_program(
        module,
        "assess_encoded",
        data={"sentence": ("x", "y"), "response": observed},
        sites={},
        parameters={
            "Mix.lex.0": weights[0],
            "Mix.lex.1": weights[1],
            "encoder.response": network,
        },
        distributions=backend,
    )
