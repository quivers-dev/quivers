"""Subsampled array priors retain shared conditioning and selected gradients."""

import pytest
import torch

from quivers.dsl import loads
from quivers.effects import subsample
from quivers.inference.trace import trace


@pytest.mark.parametrize("rows,cols", [(3, 2), (2, 2), (1, 2), (3, 1)])
@pytest.mark.parametrize("selected", [False, True])
def test_shared_mean_scores_selected_array_entries_and_all_parameter_gradients(
    rows, cols, selected
):
    source = f"""object Row : FinSet {rows}
object Col : FinSet {cols}
object One : FinSet 1
program p(mu:Real) : Row -> Row
    sample w : Row <- Normal(loc=mu,scale=1.0) [over=[Col,One]]
    return w
export p
"""
    program = loads(source).morphism
    mean = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    values = (
        torch.arange(rows * cols, dtype=torch.float64).reshape(rows, cols, 1) / 3 - 0.4
    ).requires_grad_()
    row_index = torch.tensor(sorted({0, rows - 1})) if selected else torch.arange(rows)
    col_index = torch.tensor([cols - 1]) if selected else torch.arange(cols)
    with subsample("Row", row_index), subsample("Col", col_index):
        recorded = trace(program, torch.zeros(1, 1), {"mu": mean, "w": values})
    actual = recorded.sites["w"].log_prob.sum()
    factor = rows / len(row_index) * cols / len(col_index)
    expected = (
        factor
        * torch.distributions.Normal(mean, 1.0)
        .log_prob(values[row_index][:, col_index])
        .sum()
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (mean, values), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (mean, values))
    for got, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, wanted)
    if selected and cols > 1:
        assert bool((actual_grad[1][:, :-1] == 0).all())
    # A discarded shared input would score the same entries at mean zero.
    incorrect = (
        factor
        * torch.distributions.Normal(0.0, 1.0)
        .log_prob(values[row_index][:, col_index])
        .sum()
    )
    assert not torch.isclose(actual, incorrect)
