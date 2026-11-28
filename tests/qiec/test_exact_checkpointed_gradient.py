"""Checkpoint reverse gradients against unrolled executions and cost oracles."""

from fractions import Fraction

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ExactCheckpointedGradient,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    QiecModule,
    discharge_module,
    exact_checkpointed_gradient,
)
from quivers.qiec.checkpointing import _CheckpointSweep

_SOURCE = """
define affine(z : Real, rate : Real, offset : Real) : Real !{} =
    return rate * z + offset

define switching(z : Real, p : Real) : Real !{} =
    if z > 0.0 then
        return z - p
    else
        return z + p / 2.0

define unread(z : Real) : Real !{} =
    if z > 0.0 then
        return 2.0
    else
        return 1.0

define forever(z : Real) : Real !{} =
    forever(z)

effect Increment
    apply : Real -> Real

instance increment : Increment

handler affine_step for Increment : Real -> Real with (rate : Real, offset : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    apply(v : Real) resumes 1 =>
        resume(v * rate + offset)

define authored(z : Real, rate : Real, offset : Real) : Real !{} =
    handle increment with affine_step(rate, offset) in
        perform increment.apply(z)

define residual(z : Real) : Real !{increment} =
    perform increment.apply(z)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    compiler = Compiler(
        parse(_SOURCE), module_name="scalar-replay", file_path="replay.qvr"
    )
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _affine_closed_form(
    state: Fraction, rate: Fraction, offset: Fraction, steps: int
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Solve the scalar recurrence as a polynomial of the shared parameters."""
    powers = [rate**power for power in range(steps + 1)]
    sum_powers = sum(powers[:steps], Fraction())
    state_addend = steps * powers[steps - 1] * state if steps else Fraction()
    offset_addend = offset * sum(
        (power * powers[power - 1] for power in range(1, steps)), Fraction()
    )
    return (
        powers[-1] * state + offset * sum_powers,
        (powers[-1], state_addend + offset_addend, sum_powers),
    )


@pytest.mark.parametrize("steps", (0, 1, 2, 3, 7, 16))
@pytest.mark.parametrize("rate", (Fraction(2, 3), Fraction(), Fraction(1)))
def test_checkpoint_pass_agrees_with_affine_closed_form(
    module: QiecModule, steps: int, rate: Fraction
) -> None:
    initial, offset = Fraction(7, 5), Fraction(-2, 7)
    result = exact_checkpointed_gradient(
        module, "affine", initial, (rate, offset), steps
    )
    assert (result.value, result.gradient) == _affine_closed_form(
        initial, rate, offset, steps
    )
    assert result.semantics.unrolled == ((steps,) if steps else ())
    assert result.reverse_steps == steps


def test_checkpoint_pass_agrees_with_discharged_handler_and_torch(
    module: QiecModule,
) -> None:
    steps = 11
    values = tuple(map(Fraction, (2, 3, -1)))
    discharged = discharge_module(module)
    assert discharged.obstacle("authored") is None
    source = exact_checkpointed_gradient(
        module, "authored", values[0], values[1:], steps
    )
    projected = exact_checkpointed_gradient(
        discharged.module, "authored", values[0], values[1:], steps
    )
    assert source == projected
    initial, rate, offset = inputs = tuple(
        torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
        for value in values
    )
    state = initial
    for _ in range(steps):
        state = rate * state + offset
    gradient = torch.autograd.grad(state, inputs)
    assert float(state.detach()) == pytest.approx(float(source.value), rel=1e-13)
    assert tuple(map(float, gradient)) == pytest.approx(
        tuple(map(float, source.gradient)), rel=1e-13
    )


def test_state_is_replayed_in_order_across_branch_switches(module: QiecModule) -> None:
    point, p = Fraction(4, 3), Fraction(1)
    value = point
    shifts: list[Fraction] = []
    for _ in range(9):
        assert value != 0
        scale = Fraction(-1) if value > 0 else Fraction(1, 2)
        shifts.append(scale)
        value += scale * p
    assert len(set(shifts)) == 2
    result = exact_checkpointed_gradient(module, "switching", point, (p,), 9)
    assert result.value == value
    assert result.gradient == (Fraction(1), sum(shifts, Fraction()))


@pytest.mark.parametrize("steps", (1, 2, 4, 7, 16, 33, 64, 128, 512))
def test_checkpoint_counts_match_actual_replays_and_logarithmic_storage(
    module: QiecModule, monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    observed: list[tuple[Fraction, bool, int]] = []
    evaluate = _CheckpointSweep._evaluate

    def instrumented(
        self: _CheckpointSweep, state: Fraction, *, record: bool
    ) -> tuple[Fraction, tuple[Fraction, ...]]:
        observed.append((state, record, len(self.checkpoints)))
        return evaluate(self, state, record=record)

    monkeypatch.setattr(_CheckpointSweep, "_evaluate", instrumented)
    result = exact_checkpointed_gradient(module, "affine", 0, (1, 1), steps)
    reverse = [value for value, record, _ in observed if record]
    actual_replay = len(observed) - len(reverse)
    actual_peak = max(saved for _, _, saved in observed)
    assert reverse == list(range(steps - 1, -1, -1))
    assert (
        result.replay_steps == actual_replay == sum(i.bit_count() for i in range(steps))
    )
    assert result.reverse_steps == len(reverse) == steps
    assert result.peak_saved_states == actual_peak == 1 + (steps - 1).bit_length()


def test_zero_cotangent_still_requires_a_regular_replayed_transition(
    module: QiecModule,
) -> None:
    with pytest.raises(JvpBoundaryError, match="input-dependent equality boundary"):
        exact_checkpointed_gradient(module, "unread", 0, (), 2)


def test_fuel_and_replay_work_limits_are_independent(module: QiecModule) -> None:
    with pytest.raises(JvpLimit, match="transition limit exhausted"):
        exact_checkpointed_gradient(module, "affine", 3, (1, 2), 8, max_work=19)
    completed = exact_checkpointed_gradient(module, "affine", 3, (1, 2), 8, max_work=20)
    assert completed == ExactCheckpointedGradient(
        Fraction(19), (Fraction(1), Fraction(80), Fraction(8)), 12, 8, 4
    )
    with pytest.raises(JvpLimit, match="steps"):
        exact_checkpointed_gradient(module, "forever", 3, (), 1, fuel=10)
    assert exact_checkpointed_gradient(module, "forever", 3, (), 0) == (
        ExactCheckpointedGradient(Fraction(3), (Fraction(1),), 0, 0, 0)
    )


def test_replay_requires_an_empty_residual_effect_row(module: QiecModule) -> None:
    with pytest.raises(JvpUnsupported, match="closed scalar real computation"):
        exact_checkpointed_gradient(module, "residual", 3, (), 2)
