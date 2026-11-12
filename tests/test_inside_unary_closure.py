"""The inside algorithm closes a cell under unary rules with the star ``U*``.

A cell's weights after any number of unary steps are ``c U*``, with
``U* = sum_k U^k = (I - U)^{-1}`` for a matrix ``U`` of unary rule
probabilities whose spectral radius is below one. The closure is checked
against the Neumann series summed in double precision until its terms
vanish, which is an independent statement of the same sum; the iterated
update ``c (I + U)^k`` the chart used to apply is shown to be another
number, and a cycle of unary rules with total weight one is refused.
"""

from __future__ import annotations

import pytest
import torch

from quivers.stochastic.inside import _apply_unary_closure, _unary_star


def _neumann(cell: torch.Tensor, unary: torch.Tensor) -> torch.Tensor:
    """``c sum_k U^k`` by summing the series until its terms vanish.

    Parameters
    ----------
    cell : torch.Tensor
        Shape ``(batch, N)``, probabilities.
    unary : torch.Tensor
        Shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        The closed cell.
    """
    total = cell.clone()
    term = cell.clone()
    for _ in range(2000):
        term = term @ unary
        total = total + term
        if float(term.abs().max()) < 1e-18:
            return total
    raise AssertionError("the Neumann series did not converge")


def _closure(cell: torch.Tensor, unary: torch.Tensor) -> torch.Tensor:
    """The chart's closure of a probability cell.

    Parameters
    ----------
    cell : torch.Tensor
        Shape ``(batch, N)``.
    unary : torch.Tensor
        Shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        The closed cell, as probabilities.
    """
    return _apply_unary_closure(cell.log(), _unary_star(unary.log())).exp()


def test_a_self_loop_of_one_half_doubles_the_weight() -> None:
    cell = torch.tensor([[1.0]], dtype=torch.float64)
    unary = torch.tensor([[0.5]], dtype=torch.float64)
    assert float(_closure(cell, unary)) == pytest.approx(2.0, rel=1e-12)


def test_one_rule_carries_its_probability_once() -> None:
    # A -> B with probability 1/2: B gains half of A's weight, and A
    # nothing back.
    cell = torch.tensor([[1.0, 1e-300]], dtype=torch.float64)
    unary = torch.tensor([[1e-300, 0.5], [1e-300, 1e-300]], dtype=torch.float64)
    closed = _closure(cell, unary)
    assert closed[0].tolist() == pytest.approx([1.0, 0.5], rel=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_the_closure_is_the_neumann_series(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    n = 5
    raw = torch.rand(n, n, generator=generator, dtype=torch.float64)
    # Rows summing to 0.9 keep the spectral radius at most 0.9.
    unary = 0.9 * raw / raw.sum(dim=1, keepdim=True)
    cell = torch.rand(3, n, generator=generator, dtype=torch.float64)
    closed = _closure(cell, unary)
    expected = _neumann(cell, unary)
    assert torch.allclose(closed, expected, rtol=1e-10, atol=0.0)
    # The update cell <- cell + cell U applied eight times, which is
    # c (I + U)^8, is not the closure.
    iterated = cell.clone()
    for _ in range(8):
        iterated = iterated + iterated @ unary
    assert not torch.allclose(iterated, expected, rtol=1e-2)


def test_unary_rules_of_total_weight_one_on_a_cycle_are_refused() -> None:
    unary = torch.tensor([[1e-300, 1.0], [1.0, 1e-300]], dtype=torch.float64)
    with pytest.raises(ValueError, match="spectral radius"):
        _unary_star(unary.log())


def test_the_closure_is_differentiable_in_the_rules() -> None:
    unary = torch.tensor([[0.3, 0.2], [0.1, 0.4]], dtype=torch.float64)
    log_unary = unary.log().requires_grad_(True)
    cell = torch.tensor([[1.0, 0.5]], dtype=torch.float64).log()
    total = _apply_unary_closure(cell, _unary_star(log_unary)).exp().sum()
    (grad,) = torch.autograd.grad(total, log_unary)
    # d/dU of 1^T c (I - U)^{-1} 1 is (c S)^T (S 1) with S = (I - U)^{-1},
    # times U for the log parameterization.
    star = torch.linalg.inv(torch.eye(2, dtype=torch.float64) - unary)
    c = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    expected = (c @ star).T @ (star @ torch.ones(2, 1, dtype=torch.float64)).T * unary
    assert torch.allclose(grad, expected, rtol=1e-10)
