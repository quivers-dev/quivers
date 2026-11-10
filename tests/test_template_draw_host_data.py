"""Host data read inside a program drawn from another program.

The checked elaboration types the free names a drawn program's steps
read as they are read there, so a gather by a host index inside the
drawn program types the index as integral, and the caller scores and
samples the instantiated steps.
"""

from __future__ import annotations

import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.inference import trace

SOURCE = """
object Row : FinSet 4
object Group : FinSet 2
object Val : Real 1

program part(tau : Real) : Row -> Val
    sample effect : Group <- Normal(loc=0.0, scale=tau)
    let row_effect = effect[group_idx]
    return row_effect

program caller : Row -> Val
    sample a <- part(0.7)
    observe y : Row <- Normal(loc=a, scale=1.0)
    return y

export caller
"""

EFFECT = torch.tensor([0.3, -0.5])
Y = torch.tensor([0.1, 0.2, -0.4, 1.0])
GROUP_IDX = torch.tensor([0, 1, 1, 0])


def test_a_drawn_program_gathers_by_a_host_index() -> None:
    model = loads(SOURCE).morphism
    expected = (
        D.Normal(0.0, 0.7).log_prob(EFFECT).sum()
        + D.Normal(EFFECT[GROUP_IDX], 1.0).log_prob(Y).sum()
    )
    joint = model.log_joint(
        torch.zeros(1, 1), {"a$effect": EFFECT, "y": Y, "group_idx": GROUP_IDX}
    )
    torch.testing.assert_close(joint.sum(), expected)


def test_a_drawn_program_with_a_host_index_samples() -> None:
    model = loads(SOURCE).morphism
    run = trace(model, torch.zeros(1, 1), observations={"group_idx": GROUP_IDX})
    assert run.sites["y"].value.numel() == 4
    assert run.sites["a$effect"].value.numel() == 2
