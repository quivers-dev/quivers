"""`SVI.run` draws a non-finite first objective again, and only the first.

The objective here is a counter-driven stand-in: it returns infinity for a
set number of calls and then a quadratic loss in one guide parameter. The
expected call counts and the expected parameter after each plain SGD step
(``w - lr * 2 w``) are computed by hand.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.inference import svi as svi_module
from quivers.inference.svi import SVI


class _Objective:
    def __init__(self, failures: int, fail_at_call: int | None = None) -> None:
        self.failures = failures
        self.fail_at_call = fail_at_call
        self.calls = 0

    def __call__(self, model, guide, x, observations) -> torch.Tensor:
        self.calls += 1
        if self.calls <= self.failures or self.calls == self.fail_at_call:
            return guide.weight.sum() * math.inf
        return guide.weight.square().sum()


def _runner(objective: _Objective) -> tuple[SVI, torch.nn.Linear]:
    guide = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        guide.weight.fill_(1.0)
    model = torch.nn.Identity()
    optim = torch.optim.SGD(guide.parameters(), lr=0.25)
    return SVI(model, guide, optim, objective), guide


def test_a_non_finite_first_objective_is_drawn_again_without_an_update() -> None:
    objective = _Objective(failures=7)
    runner, guide = _runner(objective)
    losses = runner.run(torch.zeros(1, 1), {}, num_steps=2)
    # Seven discarded draws, then two steps: w = 1 -> 0.5 -> 0.25.
    assert objective.calls == 9
    assert losses == pytest.approx((1.0, 0.25))
    assert float(guide.weight) == pytest.approx(0.25)


def test_the_attempt_budget_is_reported() -> None:
    assert svi_module.INITIAL_ATTEMPTS == 100
    objective = _Objective(failures=10_000)
    runner, guide = _runner(objective)
    with pytest.raises(ValueError, match="step 0: nonfinite .* after 100 draws"):
        runner.run(torch.zeros(1, 1), {}, num_steps=3)
    assert objective.calls == 100
    assert float(guide.weight) == 1.0


def test_a_later_non_finite_objective_is_not_drawn_again() -> None:
    objective = _Objective(failures=0, fail_at_call=2)
    runner, _ = _runner(objective)
    with pytest.raises(ValueError, match="step 1: nonfinite variational objective$"):
        runner.run(torch.zeros(1, 1), {}, num_steps=3)
    assert objective.calls == 2


def test_another_error_at_the_first_step_is_raised_at_once() -> None:
    class Broken(_Objective):
        def __call__(self, model, guide, x, observations) -> torch.Tensor:
            self.calls += 1
            raise ValueError("observed value outside the support")

    objective = Broken(failures=0)
    runner, _ = _runner(objective)
    with pytest.raises(ValueError, match="step 0: observed value outside"):
        runner.run(torch.zeros(1, 1), {}, num_steps=1)
    assert objective.calls == 1
