"""Matrix-valued draws preserve event axes under conditioning and batching."""

import pytest
import torch

from quivers.continuous.inline import make_fixed_normal, make_fixed_lkj_cholesky
from quivers.continuous.spaces import CholeskyFactor, Euclidean
from quivers.dsl import loads
from quivers.effects.interpreter import _event_numel
from quivers.inference.trace import trace


@pytest.mark.parametrize("dimension", [1, 3])
def test_scalar_and_vector_event_sizes_remain_coordinate_counts(dimension):
    family = make_fixed_normal(0.0, 1.0, Euclidean(name="V", dim=dimension))
    assert _event_numel(family) == dimension


def test_matrix_event_size_counts_all_carrier_entries():
    family = make_fixed_lkj_cholesky(1.5, CholeskyFactor(name="L", dim=3))
    assert _event_numel(family) == 9


@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("per_run", [False, True])
@pytest.mark.parametrize("project", [False, True])
def test_conditioned_matrix_events_preserve_shapes_and_log_density(
    runs, per_run, project
):
    source = "object K : FinSet 2\nobject Out : Real 1\nprogram prior : K -> Out\n    sample L : K <- LKJCholesky(concentration=1.5)\n"
    source += (
        "    let result = L[1, 0]\n    return result\n" if project else "    return L\n"
    )
    source += "export prior\n"
    program = loads(source).morphism
    rho = torch.linspace(-0.4, 0.6, runs) if per_run else torch.tensor([0.2])
    factors = torch.zeros(len(rho), 2, 2)
    factors[:, 0, 0] = 1
    factors[:, 1, 0] = rho
    factors[:, 1, 1] = torch.sqrt(1 - rho.square())
    supplied = factors if per_run else factors[0]
    recorded = trace(program, torch.zeros(runs, 1, dtype=torch.long), {"L": supplied})
    expected_factors = factors if per_run else factors.expand(runs, 2, 2)
    torch.testing.assert_close(recorded.sites["L"].value, expected_factors)
    reference = torch.distributions.LKJCholesky(2, 1.5).log_prob(expected_factors)
    torch.testing.assert_close(recorded.log_joint, reference)
    if project:
        torch.testing.assert_close(
            recorded.output.reshape(runs), expected_factors[:, 1, 0]
        )
    else:
        torch.testing.assert_close(recorded.output, expected_factors)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("learned", [False, True])
def test_observed_lkj_uses_intrinsic_square_axes_not_their_flattened_product(
    dimension, learned
):
    rows = 4
    argument = "alpha" if learned else "1.5"
    header = "(alpha:Tensor[Real]([Obs]))" if learned else ""
    source = f"""object Dim : FinSet {dimension}
object Obs : FinSet {rows}
program model{header} : Obs -> Obs
    observe y : Obs <- LKJCholesky(concentration={argument}) [over=[Dim,Dim]]
    return y
export model
"""
    program = loads(source).morphism
    y = torch.eye(dimension).expand(rows, dimension, dimension)
    alpha = torch.linspace(0.7, 2.0, rows, requires_grad=True)
    observations = {"y": y} | ({"alpha": alpha} if learned else {})
    recorded = trace(program, torch.zeros(1, 1), observations)
    expected = (
        torch.distributions.LKJCholesky(dimension, alpha if learned else 1.5)
        .log_prob(y)
        .sum()
    )
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    assert recorded.sites["y"].value.shape == (rows, dimension, dimension)
    if learned:
        torch.testing.assert_close(
            torch.autograd.grad(recorded.log_joint.sum(), alpha, retain_graph=True)[0],
            torch.autograd.grad(expected, alpha)[0],
        )


def test_rectangular_matrix_normal_retains_both_intrinsic_axes():
    source = """object Obs : FinSet 5
object Rows : FinSet 2
object Cols : FinSet 3
program model : Obs -> Obs
    observe y : Obs <- MatrixNormal(loc=[[0.1,0.2,0.3],[-0.1,-0.2,-0.3]],row_covariance=[[0.64,0.16],[0.16,0.53]],col_covariance=[[0.81,0.09,-0.18],[0.09,0.65,0.06],[-0.18,0.06,0.54]]) [over=[Rows,Cols]]
    return y
export model
"""
    program = loads(source).morphism
    y = torch.linspace(-0.5, 0.7, 30).reshape(5, 2, 3)
    row = torch.tensor([[0.8, 0.0], [0.2, 0.7]])
    column = torch.tensor([[0.9, 0.0, 0.0], [0.1, 0.8, 0.0], [-0.2, 0.1, 0.7]])
    loc = torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    expected = (
        torch.distributions.MultivariateNormal(
            loc.reshape(-1), scale_tril=torch.kron(row, column)
        )
        .log_prob(y.reshape(5, 6))
        .sum()
    )
    recorded = trace(program, torch.zeros(1, 1), {"y": y})
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
