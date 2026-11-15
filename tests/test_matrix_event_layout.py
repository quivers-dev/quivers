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
