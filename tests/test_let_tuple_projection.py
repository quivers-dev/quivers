"""A let indexes a tuple bound by an earlier let.

A tuple's index is a static position: ``paired[1]`` is its second
component, and a tensor component is indexed once it is bound, as
``first = pair[0]`` then ``first[1]``. Scoring, tracing, and forward
sampling all read the projection.
"""

from __future__ import annotations

import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.inference import trace

HEAD = """
object Row : FinSet 4
object Group : FinSet 2
object Val : Real 1
program p : Row -> Val
    sample effect : Group <- Normal(loc=0.0, scale=1.0)
{lets}
    observe y : Row <- Normal(loc=chosen, scale=1.0)
    return y
export p
"""

EFFECT = torch.tensor([0.3, -0.5])
Y = torch.tensor([0.1, 0.2, -0.4, 1.0])


def _expected(location: torch.Tensor) -> torch.Tensor:
    return (
        D.Normal(0.0, 1.0).log_prob(EFFECT).sum()
        + D.Normal(location, 1.0).log_prob(Y).sum()
    )


def _model(lets: str):  # type: ignore[no-untyped-def]
    return loads(HEAD.format(lets=lets)).morphism


def test_a_tuple_component_is_read_by_its_position() -> None:
    model = _model(
        "    let paired = (effect[0], effect[1])\n    let chosen = paired[1]"
    )
    joint = model.log_joint(torch.zeros(1, 1), {"effect": EFFECT, "y": Y})
    torch.testing.assert_close(joint.sum(), _expected(EFFECT[1]))


def test_a_nested_tuple_is_projected_one_position_at_a_time() -> None:
    model = _model(
        "    let nested = ((effect[0], effect[1]), 2.0)\n"
        "    let inner = nested[0]\n"
        "    let chosen = inner[0]"
    )
    joint = model.log_joint(torch.zeros(1, 1), {"effect": EFFECT, "y": Y})
    torch.testing.assert_close(joint.sum(), _expected(EFFECT[0]))


def test_a_tensor_component_is_indexed_after_its_projection() -> None:
    model = _model(
        "    let pair = (effect, 2.0)\n    let first = pair[0]\n    let chosen = first[1]"
    )
    joint = model.log_joint(torch.zeros(1, 1), {"effect": EFFECT, "y": Y})
    torch.testing.assert_close(joint.sum(), _expected(EFFECT[1]))


def test_a_program_with_a_tuple_let_samples_and_traces() -> None:
    model = _model(
        "    let paired = (effect[0], effect[1])\n    let chosen = paired[1]"
    )
    x = torch.zeros(1, 1)
    assert model.rsample(x).numel() == 4
    run = trace(model, x)
    assert "paired" not in run.sites
    values = {name: site.value for name, site in run.sites.items()}
    assert torch.isfinite(model.log_joint(x, values)).all()
