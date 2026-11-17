"""Plates reuse declared tensor event axes for draws, scores, and pullbacks."""

import math

import pytest
import scipy.stats
import torch

from quivers.continuous.inline import make_fixed_normal
from quivers.continuous.plate import PlateDraw
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference.trace import trace


def _matrix_program(rows, cols):
    return loads(f"""object Plate : FinSet 4
object R : FinSet {rows}
object C : FinSet {cols}
object Out : Real 1
program p(loc:Tensor[Real]([R,C]), U:Tensor[Real]([R,R]), V:Tensor[Real]([C,C])) : Plate -> Out
    sample M : Plate <- MatrixNormal(loc=loc,row_covariance=U,col_covariance=V) [over=[R,C]]
    return M
export p
""").morphism


@pytest.mark.parametrize("rows,cols", [(2, 3), (1, 3), (3, 1), (1, 1)])
def test_checked_matrix_plate_draws_preserve_axes_and_seeded_gaussian_cells(rows, cols):
    program = _matrix_program(rows, cols)
    observations = {
        "loc": torch.zeros(rows, cols),
        "U": torch.eye(rows),
        "V": torch.eye(cols),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        recorded = trace(program, torch.zeros(1, 1), observations)
        torch.manual_seed(71)
        expected = torch.randn(4, rows, cols)
    value = recorded.sites["M"].value
    assert value.shape == (4, rows, cols)
    torch.testing.assert_close(value, expected, atol=0, rtol=0)
    expected_score = torch.distributions.Normal(0.0, 1.0).log_prob(expected).sum()
    torch.testing.assert_close(recorded.sites["M"].log_prob.sum(), expected_score)


@pytest.mark.parametrize("rows,cols", [(2, 3), (1, 3), (3, 1), (1, 1)])
def test_matrix_plate_joint_and_all_gradients_match_independent_kronecker_law(
    rows, cols
):
    program = _matrix_program(rows, cols)
    mean = (
        torch.linspace(-0.3, 0.4, rows * cols, dtype=torch.float64)
        .reshape(rows, cols)
        .requires_grad_()
    )
    a = (
        torch.linspace(-0.2, 0.3, rows * rows, dtype=torch.float64)
        .reshape(rows, rows)
        .requires_grad_()
    )
    b = (
        torch.linspace(0.1, 0.5, cols * cols, dtype=torch.float64)
        .reshape(cols, cols)
        .requires_grad_()
    )
    u = a @ a.T + torch.eye(rows, dtype=torch.float64)
    v = b @ b.T + torch.eye(cols, dtype=torch.float64)
    point = (
        torch.linspace(-0.6, 1.2, 4 * rows * cols, dtype=torch.float64)
        .reshape(4, rows, cols)
        .requires_grad_()
    )
    recorded = trace(
        program, torch.zeros(1, 1), {"loc": mean, "U": u, "V": v, "M": point}
    )
    actual = recorded.sites["M"].log_prob.sum()
    oracle = torch.distributions.MultivariateNormal(
        mean.flatten(), covariance_matrix=torch.kron(u, v)
    )
    expected = oracle.log_prob(point.flatten(1)).sum()
    scipy_value = sum(
        scipy.stats.matrix_normal.logpdf(
            p.detach().numpy(),
            mean.detach().numpy(),
            u.detach().numpy(),
            v.detach().numpy(),
        )
        for p in point
    )
    assert float(actual.detach()) == pytest.approx(float(scipy_value), rel=1e-11)
    torch.testing.assert_close(actual, expected)
    parameters = (mean, a, b, point)
    got = torch.autograd.grad(actual, parameters, retain_graph=True)
    wanted = torch.autograd.grad(expected, parameters)
    for x, y in zip(got, wanted, strict=True):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("event", [(2, 3), (1, 3), (3, 1), (1, 1), (1,), ()])
def test_multiple_plate_axes_and_pullback_keep_the_explicit_event_contract(event):
    carrier = Euclidean(name="event", dim=math.prod(event), intrinsic_shape=event)
    if len(event) == 2:
        family = _matrix_program(*event).steps[0].morphism.family
        x = (
            torch.cat(
                (
                    torch.zeros(math.prod(event)),
                    torch.eye(event[0]).flatten(),
                    torch.eye(event[1]).flatten(),
                )
            )
            .double()
            .reshape(1, -1)
        )
    else:
        family = make_fixed_normal(0.0, 1.0, carrier)
        x = torch.zeros(1, 1, dtype=torch.float64)
    plate = PlateDraw(6, family, index_shape=(2, 3))
    assert plate._per_row_shape == event
    values = (
        torch.linspace(-0.5, 0.8, 6 * math.prod(event), dtype=torch.float64)
        .reshape(2, 3, *event)
        .requires_grad_()
    )
    # Each declared array cell is scored once, irrespective of event rank.
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(values).sum()
    actual = plate.log_prob(x, values).sum()
    torch.testing.assert_close(actual, expected)
    index0, index1 = torch.tensor([1]), torch.tensor([0, 2])
    selected = plate.log_prob_rows(x, values, index0, index1).sum()
    oracle = (
        torch.distributions.Normal(0.0, 1.0).log_prob(values[index0][:, index1]).sum()
    )
    torch.testing.assert_close(selected, oracle)
    got = torch.autograd.grad(selected, values, retain_graph=True)[0]
    wanted = torch.autograd.grad(oracle, values)[0]
    torch.testing.assert_close(got, wanted)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79)
        draw = plate.rsample(x)
    assert draw.shape == (2, 3, *event)
    assert plate.canonical_latent(values.flatten()).shape == (2, 3, *event)
