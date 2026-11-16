"""Independent value and differential oracles for closed-weight mixture sums."""

import pytest
import torch
from quivers.continuous._weighted_logsumexp import weighted_logsumexp


def test_interior_reverse_forward_and_second_derivatives():
    w = torch.tensor([0.3, 0.8, 1.2], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([-2.0, 0.3, 1.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(weighted_logsumexp, (w, s))
    assert torch.autograd.gradgradcheck(weighted_logsumexp, (w, s))

    def direct(a, b):
        return (a * b.exp()).sum(-1).log()

    for operation in [torch.func.jacfwd, torch.func.jacrev, torch.func.hessian]:
        torch.testing.assert_close(
            operation(weighted_logsumexp, argnums=(0, 1))(w, s),
            operation(direct, argnums=(0, 1))(w, s),
        )


def test_zero_weight_one_sided_first_second_and_mixed_derivatives():
    w = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([0.7, -0.3], dtype=torch.float64, requires_grad=True)

    def direct(a, b):
        return (a * b.exp()).sum(-1).log()

    torch.testing.assert_close(
        torch.func.jacrev(weighted_logsumexp, argnums=(0, 1))(w, s),
        torch.func.jacrev(direct, argnums=(0, 1))(w, s),
    )
    torch.testing.assert_close(
        torch.func.hessian(weighted_logsumexp, argnums=(0, 1))(w, s),
        torch.func.hessian(direct, argnums=(0, 1))(w, s),
    )
    _, tangent = torch.func.jvp(
        weighted_logsumexp,
        (w, s),
        (torch.tensor([1.0, 0.0], dtype=torch.float64), torch.zeros_like(s)),
    )
    assert tangent.item() == pytest.approx(torch.exp(torch.tensor(1.0)).item())


def test_zero_weight_extreme_score_does_not_underflow_selected_component():
    w = torch.tensor([0.0, 1.0], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([2000.0, -1000.0], dtype=torch.float64, requires_grad=True)
    value = weighted_logsumexp(w, s)
    assert value.item() == -1000.0
    gw, gs = torch.autograd.grad(value, (w, s))
    assert torch.isposinf(gw[0])
    torch.testing.assert_close(gs, torch.tensor([0.0, 1.0], dtype=torch.float64))
    _, tangent = torch.func.jvp(
        weighted_logsumexp,
        (w, s),
        (torch.zeros_like(w), torch.tensor([0.0, 1.0], dtype=torch.float64)),
    )
    assert tangent.item() == 1.0


def test_all_negative_infinity_branch_has_safe_inactive_gradients():
    w = torch.tensor([0.3, 0.7], dtype=torch.float64, requires_grad=True)
    s = torch.tensor([-torch.inf, -torch.inf], dtype=torch.float64, requires_grad=True)
    zero = weighted_logsumexp(w, s)
    value = torch.where(torch.tensor(False), zero, w.sum().log())
    gw, gs = torch.autograd.grad(value, (w, s))
    torch.testing.assert_close(gw, torch.ones_like(w))
    torch.testing.assert_close(gs, torch.zeros_like(s))


def test_broadcast_nonfinal_axes_and_vmap():
    w = torch.tensor([[0.0, 1.0], [0.3, 0.7]], dtype=torch.float64)
    s = torch.tensor([0.2, 0.8], dtype=torch.float64)
    expected = (w * s.exp()).sum(-1).log()
    torch.testing.assert_close(
        torch.func.vmap(weighted_logsumexp, in_dims=(0, None))(w, s), expected
    )
    torch.testing.assert_close(weighted_logsumexp(w.T, s[:, None], dim=0), expected)


def test_zero_total_extreme_inactive_branch_has_finite_hessian():
    s = torch.tensor([2000.0, -1000.0], dtype=torch.float64)

    def inactive(w):
        score = weighted_logsumexp(w, s)
        return torch.where(torch.tensor(False), score, w.square().sum())

    w = torch.zeros(2, dtype=torch.float64)
    torch.testing.assert_close(
        torch.func.hessian(inactive)(w), 2 * torch.eye(2, dtype=torch.float64)
    )
