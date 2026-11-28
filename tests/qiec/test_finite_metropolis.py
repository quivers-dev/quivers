"""Check finite-step Metropolis laws against closed forms and proposal histories."""

from fractions import Fraction
from itertools import product

import pytest
import torch

from quivers.dsl import loads
from quivers.qiec import ExactJvp, exact_jvp, finite_metropolis_jvp
from quivers.qiec.differentiation import JvpBoundaryError, JvpLimit
from quivers.qiec.module import QiecModule


def _mass(value, tangent=0):
    return ExactJvp(Fraction(value), Fraction(tangent))


@pytest.mark.parametrize("steps", (0, 1, 2, 8, 24))
def test_two_state_law_follows_its_finite_step_closed_form(steps):
    module = loads("""
define target_ready(t : Real) : Real !{} =
    return t

define initial_ready(t : Real) : Real !{} =
    return 0.25 + t / 2.0
""").qiec
    assert isinstance(module, QiecModule)
    t = Fraction(1, 4)
    steady = t / (1 + t)
    steady_tangent = 1 / (1 + t) ** 2
    initial = exact_jvp(module, "initial_ready", (t,), (1,))
    target = exact_jvp(module, "target_ready", (t,), (1,))
    complement = _mass(1 - initial.value, -initial.tangent)
    proposal = (
        (_mass(Fraction(3, 4)), _mass(Fraction(1, 4))),
        (_mass(Fraction(1, 2)), _mass(Fraction(1, 2))),
    )
    result = finite_metropolis_jvp(
        (complement, initial), (_mass(1), target), proposal, steps=steps
    )
    decay = (1 - t) / 2
    correction = (initial.value - steady) * decay**steps
    correction_tangent = (initial.tangent - steady_tangent) * decay**steps
    if steps:
        correction_tangent -= (
            (initial.value - steady) * steps * decay ** (steps - 1) / 2
        )
    assert result[1] == _mass(steady + correction, steady_tangent + correction_tangent)
    assert result[0] == _mass(1 - result[1].value, -result[1].tangent)
    assert result[1].tangent != steady_tangent


def test_joint_proposal_acceptance_histories_include_rejections_and_live_proposal():
    initial = (
        _mass(Fraction(1, 4), Fraction(1, 8)),
        _mass(Fraction(1, 4)),
        _mass(Fraction(1, 2), Fraction(-1, 8)),
    )
    target = (
        _mass(1),
        _mass(Fraction(2, 3), Fraction(1, 7)),
        _mass(Fraction(4, 5), Fraction(-1, 8)),
    )
    proposal = (
        (
            _mass(Fraction(1, 5), Fraction(-1, 15)),
            _mass(Fraction(1, 2), Fraction(1, 15)),
            _mass(Fraction(3, 10)),
        ),
        (_mass(Fraction(1, 4)), _mass(Fraction(1, 4)), _mass(Fraction(1, 2))),
        (
            _mass(Fraction(1, 2)),
            _mass(Fraction(1, 4), Fraction(-1, 12)),
            _mass(Fraction(1, 4), Fraction(1, 12)),
        ),
    )
    steps = 3
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def point(cell):
        return theta * float(cell.tangent) + float(cell.value)

    a = [point(cell) for cell in target]
    q = [[point(cell) for cell in row] for row in proposal]
    final = [theta * 0 for _ in initial]
    actions = tuple(product(range(3), (False, True)))
    for start, mass in enumerate(initial):
        for history in product(actions, repeat=steps):
            current = start
            probability = point(mass)
            for proposed, take in history:
                accept = (
                    a[proposed]
                    * q[proposed][current]
                    / (a[current] * q[current][proposed])
                ).clamp(max=1)
                probability = (
                    probability
                    * q[current][proposed]
                    * (accept if take else 1 - accept)
                )
                if take:
                    current = proposed
            final[current] = final[current] + probability

    checked = finite_metropolis_jvp(initial, target, proposal, steps=steps)
    frozen_target = tuple(_mass(cell.value) for cell in target)
    frozen_proposal = tuple(
        tuple(_mass(cell.value) for cell in row) for row in proposal
    )
    derivative_without_acceptance_or_proposal = finite_metropolis_jvp(
        initial, frozen_target, frozen_proposal, steps=steps
    )
    assert sum(cell.value for cell in checked) == 1
    assert sum(cell.tangent for cell in checked) == 0
    assert any(
        a.tangent != b.tangent
        for a, b in zip(checked, derivative_without_acceptance_or_proposal, strict=True)
    )
    for expected, cell in zip(final, checked, strict=True):
        (gradient,) = torch.autograd.grad(expected, theta, retain_graph=True)
        torch.testing.assert_close(
            expected.detach(), expected.new_tensor(float(cell.value))
        )
        torch.testing.assert_close(gradient, gradient.new_tensor(float(cell.tangent)))


def test_acceptance_kink_and_work_limit_are_explicit():
    initial = (_mass(1), _mass(0))
    target = (_mass(1), _mass(Fraction(1, 2), 1))
    proposal = (
        (_mass(Fraction(3, 4)), _mass(Fraction(1, 4))),
        (_mass(Fraction(1, 2)), _mass(Fraction(1, 2))),
    )
    assert finite_metropolis_jvp(initial, target, proposal, steps=0) == initial
    with pytest.raises(JvpBoundaryError, match="unequal directional slopes"):
        finite_metropolis_jvp(initial, target, proposal, steps=1)
    with pytest.raises(JvpLimit, match="16 work positions"):
        finite_metropolis_jvp(initial, target, proposal, steps=1, max_work=15)
    assert finite_metropolis_jvp(
        initial, (_mass(1, 2), _mass(Fraction(1, 2), 1)), proposal, steps=1
    ) == (_mass(Fraction(3, 4)), _mass(Fraction(1, 4)))


def test_zero_proposal_carries_no_tangent_and_can_make_the_chain_reducible():
    initial = (_mass(Fraction(1, 3)), _mass(Fraction(2, 3)))
    target = (_mass(1), _mass(1))
    absorbing = ((_mass(1), _mass(0)), (_mass(0), _mass(1)))
    assert finite_metropolis_jvp(initial, target, absorbing, steps=3) == initial
    with pytest.raises(ValueError, match="nonnegative mass"):
        finite_metropolis_jvp(
            initial, target, ((_mass(1, -1), _mass(0, 1)), absorbing[1]), steps=1
        )
