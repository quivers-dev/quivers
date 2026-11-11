"""An observe drawn over more `over` axes than its family's event rank.

Each row of the plate holds an independent array over the leading axes
of one variate on the trailing ones, as a sample does: forward sampling
draws a value of the checked shape, and the joint scores it unchanged.
"""

from __future__ import annotations

import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.inference import trace

HEAD = """
object N : FinSet 2
object B : FinSet 3
object C : FinSet 2
object K : FinSet 4
object Val : Real 1
program p : N -> Val
{body}
    return y
export p
"""


def _program(body: str):  # type: ignore[no-untyped-def]
    return loads(HEAD.format(body=body))


def _checked_shape(program) -> tuple[int, ...]:  # type: ignore[no-untyped-def]
    entry = next(e for e in program.qiec.entries if e.name == "p")
    site = next(s for s in entry.sites if s.name == "y")
    return tuple(int(axis.size.value) for axis in (*site.batch, *site.event))


def test_a_scalar_observe_over_two_axes_draws_its_checked_shape() -> None:
    program = _program(
        "    sample mu <- Normal(loc=0.0, scale=1.0)\n"
        "    observe y : N <- Normal(loc=mu, scale=1.0) [over=[B, C]]"
    )
    assert _checked_shape(program) == (2, 3, 2)
    x = torch.zeros(1, 1)
    assert tuple(program.morphism.rsample(x).shape) == (2, 3, 2)
    run = trace(program.morphism, x)
    assert tuple(run.sites["y"].value.shape) == (2, 3, 2)
    mu, y = torch.tensor([0.2]), torch.linspace(-1.0, 1.0, 12).reshape(2, 3, 2)
    expected = (
        D.Normal(0.0, 1.0).log_prob(mu).sum() + D.Normal(mu, 1.0).log_prob(y).sum()
    )
    joint = program.morphism.log_joint(x, {"mu": mu, "y": y})
    torch.testing.assert_close(joint.sum(), expected)


def test_each_row_draws_its_array_at_its_own_parameters() -> None:
    program = _program(
        "    sample z : N <- Normal(loc=0.0, scale=10.0)\n"
        "    observe y : N <- Normal(loc=z, scale=0.01) [over=[B, C]]"
    )
    torch.manual_seed(0)
    run = trace(program.morphism, torch.zeros(1, 1))
    z, y = run.sites["z"].value.reshape(2), run.sites["y"].value
    assert tuple(y.shape) == (2, 3, 2)
    torch.testing.assert_close(y, z.reshape(2, 1, 1).expand(2, 3, 2), atol=0.1, rtol=0)


def test_a_vector_observe_over_an_extra_axis_draws_one_simplex_per_entry() -> None:
    program = _program(
        "    observe y : N <- Dirichlet(concentration=2.0) [over=[B, K]]"
    )
    assert _checked_shape(program) == (2, 3, 4)
    draw = program.morphism.rsample(torch.zeros(1, 1))
    assert tuple(draw.shape) == (2, 3, 4)
    torch.testing.assert_close(draw.sum(-1), torch.ones(2, 3))
    expected = D.Dirichlet(torch.full((4,), 2.0)).log_prob(draw).sum()
    joint = program.morphism.log_joint(torch.zeros(1, 1), {"y": draw})
    torch.testing.assert_close(joint.sum(), expected)
