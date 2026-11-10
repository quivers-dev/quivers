"""An ungrouped marginalization's observation reads a latent-dependent parameter beside a literal.

At each atom of the latent, a parameter that reads the latent, such as
``mu[z]`` or ``real(z)``, is one value shared by every row of the
observation's plate, as a literal beside it is. The block scores the
mixture over the atoms.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads

SOURCE = """
object Step : FinSet 3
object K : FinSet 4
object Val : Real 1

program p : Step -> Val
    sample mu : K <- Normal(loc=0.0, scale=1.0)
    sample prior <- Dirichlet(concentration=1.0) [over=K]
    marginalize z : K <- Categorical(probs=prior) [reduction=logsumexp]
        let value = real(z)
        observe y : Step <- Normal(loc={loc}, scale={scale})
    return mu

export p
"""

MU = torch.tensor([-1.0, 0.0, 0.5, 2.0])
PRIOR = torch.tensor([0.1, 0.2, 0.3, 0.4])
Y = torch.tensor([0.4, -1.2, 2.1])


def _expected(location: list[float], scale: float) -> torch.Tensor:
    per_atom = torch.stack(
        [
            PRIOR[z].log() + D.Normal(location[z], scale).log_prob(Y).sum()
            for z in range(4)
        ]
    )
    return (
        D.Normal(0.0, 1.0).log_prob(MU).sum()
        + D.Dirichlet(torch.ones(4)).log_prob(PRIOR)
        + torch.logsumexp(per_atom, 0)
    )


@pytest.mark.parametrize("mu_shape", [(4,), (1, 4)])
def test_a_gathered_location_beside_a_literal_scale(mu_shape: tuple[int, ...]) -> None:
    model = loads(SOURCE.format(loc="mu[z]", scale="1.0")).morphism
    joint = model.log_joint(
        torch.zeros(1, 1), {"mu": MU.reshape(mu_shape), "prior": PRIOR, "y": Y}
    )
    torch.testing.assert_close(joint.sum(), _expected(MU.tolist(), 1.0))


def test_a_latent_value_location_beside_a_literal_scale() -> None:
    model = loads(SOURCE.format(loc="value", scale="1.5")).morphism
    joint = model.log_joint(torch.zeros(1, 1), {"mu": MU, "prior": PRIOR, "y": Y})
    torch.testing.assert_close(joint.sum(), _expected([0.0, 1.0, 2.0, 3.0], 1.5))
