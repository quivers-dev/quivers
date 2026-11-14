"""Built-in handler models retain live state without sharing fresh defaults."""

import didactic.api as dx
import pytest
import torch

from quivers.qiec.builtins import (
    Aborted,
    ScoreAccumulator,
    StateCell,
    TraceRecorder,
    WeightAccumulator,
)


@pytest.mark.parametrize("model", [ScoreAccumulator, WeightAccumulator, StateCell])
def test_handler_cells_retain_tensor_identity_and_gradients(
    model: type[ScoreAccumulator] | type[WeightAccumulator] | type[StateCell],
) -> None:
    """Record updates retain a shared cell and its differentiable value.

    Parameters
    ----------
    model
        The handler state model under test.
    """
    leaf = torch.tensor(2.0, requires_grad=True)
    contents = [leaf.square()]
    state = model(contents=contents)
    copied = state.with_()
    assert state.contents is contents
    assert copied.contents is contents
    assert state.contents[0] is contents[0]
    contents[0].backward()
    assert leaf.grad == 4.0
    contents[0] = torch.tensor(7.0)
    assert copied.contents[0] is contents[0]
    if isinstance(state, StateCell):
        assert state.value is contents[0]
    else:
        assert state.total is contents[0]
        assert state.contributions is not model(contents=[0]).contributions


@pytest.mark.parametrize("model", [ScoreAccumulator, WeightAccumulator, StateCell])
@pytest.mark.parametrize("contents", [[], [1, 2]])
def test_handler_cells_reject_missing_or_ambiguous_state(
    model: type[ScoreAccumulator] | type[WeightAccumulator] | type[StateCell],
    contents: list[int],
) -> None:
    """State cells require exactly one value.

    Parameters
    ----------
    model
        The handler state model under test.
    contents
        A cell with an invalid number of stored values.
    """
    with pytest.raises(dx.ValidationError, match="exactly one"):
        model(contents=contents)


def test_trace_recorders_have_independent_event_lists() -> None:
    """Fresh collectors own distinct lists, while updates retain a collector."""
    first = TraceRecorder()
    second = TraceRecorder()
    assert first.events is not second.events
    assert first.with_().events is first.events


def test_aborted_payload_retains_live_tensor_identity() -> None:
    """Abort answers retain their payload rather than copying its tensor."""
    payload = torch.tensor([1.0, 2.0], requires_grad=True)
    answer = Aborted(error=payload)
    assert answer.error is payload
    assert answer.with_().error is payload
    assert answer != Aborted(error=payload.clone())
