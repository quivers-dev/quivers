"""Run the original packed contextual model from authored QVR likelihood code."""

import didactic.api as dx
import pytest
import torch
from torch import nn

from quivers.core.relations import RelationLimitError
from quivers.dsl import Compiler, module_to_source, parse
from quivers.qiec import (
    AttachmentId,
    ContextualEditRuntimeProvider,
    CoreRuntimeProvider,
    ExecutionFailure,
    FixedContextualEditRequest,
    QiecModule,
    RuntimeConfiguration,
    discharge_module,
    dumps,
    loads,
    run_named,
)
from quivers.transpile import project_for_target
from tests.test_contextual_edit import _BatchedInputContext, _channel, _enumerate_all

_MODULE = "checked_contextual_edit_test"
_CHANNEL = AttachmentId.derive(_MODULE, "conditional_edit", "scorer")


def _request(**changes):
    options = dict(
        channel=_CHANNEL,
        input_alphabet=("inputs", 3),
        output_alphabet=("outputs", 2),
        inputs=((0, 1), (), (0, 2), (1,), (1,)),
        outputs=((1,), (1,), (1,), (0, 1), (0, 1, 0, 1)),
        max_insertions=1,
        max_states=300,
        max_work=1000,
    )
    return FixedContextualEditRequest(**(options | changes))


def _native(request, *, width=None):
    width = len(request.inputs) if width is None else width
    result = f"Tensor[LogWeight]([{width}])"
    handler = request.handler_name
    source = f"""instance batch : Compute[Unit, {result}]
handler {handler} for Compute[Unit, {result}] : {result} -> {result} [coverage=total, forwards=none, implementation=foreign]
    apply resumes 1

define observe : {result} !{{}} =
    handle batch with {handler} in
        let value <- perform batch.apply(unit)
        return value
"""
    compiler = Compiler(parse(module_to_source(parse(source))), module_name=_MODULE)
    compiler.compile()
    module = loads(dumps(compiler.qiec_module))
    assert isinstance(module, QiecModule)
    return module


def _run(module, provider):
    return run_named(
        module,
        "observe",
        runtime=RuntimeConfiguration(
            selections=(), providers=(CoreRuntimeProvider(), provider)
        ),
    )


def test_packed_native_score_and_gradients_match_complete_paths_after_update():
    scorer = _BatchedInputContext()
    request = _request()
    wire = dumps(request)
    assert isinstance(request := loads(wire), FixedContextualEditRequest)
    assert dumps(request) == wire
    channel = _channel(scorer)
    provider = ContextualEditRuntimeProvider(request, {_CHANNEL: channel})
    module = _native(request)
    assert scorer.calls == 0
    original = tuple(scorer.parameters())
    assert all(a is b for a, b in zip(original, channel.parameters(), strict=True))
    optimizer = torch.optim.SGD(original, lr=0.2)
    previous = None
    for epoch in range(2):
        result = _run(module, provider)
        assert result.result_type == request.result_type
        actual = result.value
        assert isinstance(actual, torch.Tensor)
        assert scorer.calls == 2 * epoch + 1
        assert actual.shape == (5,)
        assert actual[-1].item() == -torch.inf
        assert actual[0].item() != actual[2].item()
        logits = scorer(request.inputs)
        references = [
            _enumerate_all(logits, source, i, 2, 1)
            for i, source in enumerate(request.inputs)
        ]
        assert request.outputs[-1] not in references[-1]
        expected = torch.stack(
            [
                row[target].log()
                for row, target in zip(
                    references[:-1], request.outputs[:-1], strict=True
                )
            ]
        )
        torch.testing.assert_close(actual[:-1], expected, atol=3e-14, rtol=3e-14)
        for gradient in torch.autograd.grad(actual[-1], original, retain_graph=True):
            assert torch.count_nonzero(gradient).item() == 0
        expected_gradients = torch.autograd.grad(-expected.sum(), original)
        optimizer.zero_grad()
        (-actual[:-1].sum()).backward()
        for param, gradient in zip(original, expected_gradients, strict=True):
            torch.testing.assert_close(param.grad, gradient, atol=3e-14, rtol=3e-14)
        if previous is not None:
            assert not torch.equal(actual, previous)
        previous = actual.detach()
        optimizer.step()

    discharged = discharge_module(module)
    for target in ("stan", "bugs", "jags"):
        assert any(
            item.computation == "observe" and request.handler_name in item.reason
            for item in project_for_target(module, discharged, target).obstacles
        )


def test_exact_owner_batch_type_and_resource_refusals_before_model_call():
    scorer = _BatchedInputContext()
    channel = _channel(scorer)
    request = _request()
    with pytest.raises(ValueError, match="missing original contextual edit"):
        ContextualEditRuntimeProvider(request, {})
    changed_alphabet = request.with_(input_alphabet=("same_size_other_tape", 3))
    with pytest.raises(ValueError, match="different nominal tapes"):
        ContextualEditRuntimeProvider(changed_alphabet, {_CHANNEL: channel})
    with pytest.raises(ValueError, match="insertion bound"):
        ContextualEditRuntimeProvider(
            request.with_(max_insertions=0), {_CHANNEL: channel}
        )
    with pytest.raises(RelationLimitError, match="action cells"):
        ContextualEditRuntimeProvider(request.with_(max_work=0), {_CHANNEL: channel})
    with pytest.raises(RelationLimitError, match="max_states"):
        ContextualEditRuntimeProvider(request.with_(max_states=1), {_CHANNEL: channel})
    with pytest.raises(dx.ValidationError, match="paired observations"):
        request.with_(outputs=request.outputs[:-1])
    provider = ContextualEditRuntimeProvider(request, {_CHANNEL: channel})
    assert request.with_(max_work=1002).handler_name == request.handler_name
    with pytest.raises(ExecutionFailure, match="incompatible fixed relation handler"):
        _run(_native(request, width=4), provider)

    other = AttachmentId.derive(_MODULE, "other_model", "scorer")
    substitute = request.with_(channel=other)
    assert substitute.result_type == request.result_type
    assert substitute.handler_name != request.handler_name
    wrong = ContextualEditRuntimeProvider(substitute, {other: channel})
    with pytest.raises(ExecutionFailure, match="unknown fixed relation handler"):
        _run(_native(request), wrong)
    assert scorer.calls == 0


def test_native_log_shape_and_extreme_finite_weight():
    request = _request(
        inputs=((0,),),
        outputs=((0,),),
        output_alphabet=("outputs", 1),
        max_insertions=0,
    )
    scorer = _BatchedInputContext(insertion_bound=0, alphabet_size=1)
    with torch.no_grad():
        scorer.bias.copy_(scorer.bias.new_tensor((0, 0, 0, -1000)))
        scorer.context.zero_()
    channel = _channel(scorer, bound=0, output_size=1)
    module = _native(request)
    result = _run(
        module, ContextualEditRuntimeProvider(request, {_CHANNEL: channel})
    ).value
    assert isinstance(result, torch.Tensor)
    assert result.tolist() == [-1000.0]
    assert torch.autograd.grad(result.sum(), scorer.bias)[0].tolist() == [
        -1.0,
        0.0,
        0.0,
        1.0,
    ]
    wrong_shape = _channel(nn.Identity(), bound=0, output_size=1)
    with pytest.raises(ExecutionFailure, match="floating tensor of shape"):
        _run(module, ContextualEditRuntimeProvider(request, {_CHANNEL: wrong_shape}))
