"""Declared matrix events survive flat inference coordinates and bijections."""

import math

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, HMCKernel, MCMC
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace


def _matrix_program(rows, columns, plate):
    location = [[0.0] * columns for _ in range(rows)]
    row_covariance = torch.eye(rows).tolist()
    col_covariance = torch.eye(columns).tolist()
    binding = " : Batch" if plate else ""
    return loads(f"""
object Rows : FinSet {rows}
object Columns : FinSet {columns}
object Batch : FinSet 2
object One : FinSet 1
program matrices : One -> One
    sample value{binding} <- MatrixNormal(loc={location}, row_covariance={row_covariance}, col_covariance={col_covariance}) [over=[Rows, Columns]]
    return value
export matrices
""").morphism


@pytest.mark.parametrize("rows,columns", [(2, 3), (1, 3), (3, 1), (1, 1)])
@pytest.mark.parametrize("plate", [False, True])
def test_matrix_event_flat_roundtrip_joint_and_all_coordinate_gradients(
    rows, columns, plate
):
    model = _matrix_program(rows, columns, plate)
    registry = LatentRegistry.from_model(model, set())
    event_shape = (2, rows, columns) if plate else (rows, columns)
    assert registry["value"].constrained_shape == event_shape
    assert registry["value"].event_shape == (rows, columns)
    vector = torch.linspace(
        -0.4, 0.7, math.prod(event_shape), dtype=torch.float32, requires_grad=True
    )
    unconstrained = registry.unflatten_unconstrained(vector)
    constrained, jacobians = registry.to_constrained(unconstrained)
    assert constrained["value"].shape == event_shape
    inverse, inverse_jacobians = registry.to_unconstrained(constrained)
    torch.testing.assert_close(registry.flatten_unconstrained(inverse), vector)
    assert (
        jacobians["value"].sum().item() == inverse_jacobians["value"].sum().item() == 0
    )
    value = registry.trace_values(constrained, batch_size=1)
    actual = model.log_joint(torch.zeros(1, 1), value).sum()
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(vector).sum()
    torch.testing.assert_close(actual, expected)
    (gradient,) = torch.autograd.grad(actual, vector)
    torch.testing.assert_close(gradient, -vector)
    recorded = trace(model, torch.zeros(1, 1))
    traced_registry = LatentRegistry.from_trace(model, recorded)
    assert traced_registry["value"].constrained_shape == event_shape
    guide = AutoNormalGuide(model, observed_names=set())
    draw = registry.trace_row(guide.rsample(torch.zeros(1, 1)), row=0)
    assert draw["value"].shape == event_shape


@pytest.mark.parametrize(
    "rows,columns,plate", [(2, 3, False), (1, 1, False), (2, 3, True)]
)
def test_hmc_retains_matrix_events_in_actual_posterior_records(rows, columns, plate):
    model = _matrix_program(rows, columns, plate)
    fitted = MCMC(
        HMCKernel(step_size=0.1, num_steps=2),
        num_warmup=2,
        num_samples=3,
        num_chains=1,
        seed=29,
    ).run(model, torch.zeros(1, 1), {})
    event_shape = (2, rows, columns) if plate else (rows, columns)
    assert fitted.samples["value"].shape == (1, 3, *event_shape)
    assert torch.isfinite(fitted.log_densities).all()


@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("rows,columns", [(1, 3), (3, 1), (1, 1), (2, 3)])
def test_declared_matrix_events_distinguish_shared_and_per_run_values(
    runs, rows, columns
):
    model = _matrix_program(rows, columns, False)
    shared = (
        torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns) / 7
    )
    per_run = shared.expand(runs, rows, columns).clone()
    per_run += torch.arange(runs, dtype=torch.float32).reshape(runs, 1, 1) / 3
    inputs = torch.zeros(runs, 1)
    for supplied, expected in ((shared, shared.expand_as(per_run)), (per_run, per_run)):
        recorded = trace(model, inputs, {"value": supplied})
        torch.testing.assert_close(recorded.sites["value"].value, expected)
        oracle = torch.distributions.Normal(0.0, 1.0).log_prob(expected).sum((-2, -1))
        torch.testing.assert_close(recorded.log_joint, oracle)
