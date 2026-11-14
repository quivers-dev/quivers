"""Static wrapper reuse preserves each run's observations and autograd graph."""

import torch

from quivers.dsl import loads
from quivers.effects import scale


def test_repeated_runs_keep_fresh_observations_gradients_and_handler_state():
    program = loads("""
object Row : FinSet 3
object Out : Real 1
program model : Row -> Out
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return mu
export model
""").morphism
    x = torch.zeros(1, 1, dtype=torch.float64)
    for value, observations, factor in [
        (0.2, [1.0, 2.0, 3.0], 1.0),
        (-0.4, [-1.0, 0.0, 0.5], 2.0),
        (0.7, [0.1, 0.2, 0.3], 0.5),
    ]:
        mu = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([observations], dtype=torch.float64)
        with scale(factor):
            actual = program.log_joint(x, {"mu": mu, "y": y}).sum()
        expected = factor * (
            torch.distributions.Normal(0.0, 1.0).log_prob(mu).sum()
            + torch.distributions.Normal(mu, 1.0).log_prob(y).sum()
        )
        torch.testing.assert_close(actual, expected)
        (gradient,) = torch.autograd.grad(actual, mu)
        torch.testing.assert_close(
            gradient, factor * (y.sum().reshape(1) - 4 * mu.detach())
        )
