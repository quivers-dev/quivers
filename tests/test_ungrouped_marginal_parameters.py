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


MIXED = """
object Step : FinSet 3
object K : FinSet 4
object Val : Real 1

program p : Step -> Val
    sample r <- HalfNormal(scale=1.0)
    sample prior <- Dirichlet(concentration=1.0) [over=K]
    marginalize z : K <- Categorical(probs=prior) [reduction=logsumexp]
        let shift = real(z)
        let spread = real(z) + 0.5
        observe y : Step <- Normal(loc={loc}, scale={scale})
    return r

export p
"""

R = torch.tensor(0.7)

#: Each source of a parameter at atom ``z``: its expression and its value.
LOCATIONS = {
    "literal": ("0.25", lambda z: 0.25),
    "sampled-scalar": ("r", lambda z: float(R)),
    "let-over-latent": ("shift", lambda z: float(z)),
}
SCALES = {
    "literal": ("1.5", lambda z: 1.5),
    "sampled-scalar": ("r", lambda z: float(R)),
    "let-over-latent": ("spread", lambda z: z + 0.5),
}


def _mixed_expected(loc, scale) -> torch.Tensor:  # type: ignore[no-untyped-def]
    per_atom = torch.stack(
        [
            PRIOR[z].log() + D.Normal(loc(z), scale(z)).log_prob(Y).sum()
            for z in range(4)
        ]
    )
    return (
        D.HalfNormal(1.0).log_prob(R)
        + D.Dirichlet(torch.ones(4)).log_prob(PRIOR)
        + torch.logsumexp(per_atom, 0)
    )


@pytest.mark.parametrize("scale", sorted(SCALES))
@pytest.mark.parametrize("loc", sorted(LOCATIONS))
def test_every_parameter_source_scores_the_mixture(loc: str, scale: str) -> None:
    loc_expr, loc_value = LOCATIONS[loc]
    scale_expr, scale_value = SCALES[scale]
    model = loads(MIXED.format(loc=loc_expr, scale=scale_expr)).morphism
    x = torch.zeros(1, 1)
    sites = {"r": R.reshape(1), "prior": PRIOR, "y": Y}
    joint = model.log_joint(x, sites)
    torch.testing.assert_close(joint.sum(), _mixed_expected(loc_value, scale_value))
    draw = model.rsample(x, observations={"y": Y})
    assert torch.isfinite(draw).all()


def test_a_let_over_the_latent_beside_a_sampled_scale_scores_the_mixture() -> None:
    model = loads(
        SOURCE.replace(
            "    sample mu", "    sample r <- HalfNormal(scale=1.0)\n    sample mu"
        ).format(loc="value", scale="r")
    ).morphism
    r = torch.tensor([0.7])
    joint = model.log_joint(
        torch.zeros(1, 1), {"mu": MU, "prior": PRIOR, "y": Y, "r": r}
    )
    per_atom = torch.stack(
        [PRIOR[z].log() + D.Normal(float(z), 0.7).log_prob(Y).sum() for z in range(4)]
    )
    expected = (
        D.Normal(0.0, 1.0).log_prob(MU).sum()
        + D.HalfNormal(1.0).log_prob(r).sum()
        + D.Dirichlet(torch.ones(4)).log_prob(PRIOR)
        + torch.logsumexp(per_atom, 0)
    )
    torch.testing.assert_close(joint.sum(), expected)
    torch.testing.assert_close(joint.sum(), torch.tensor(-14.2548), atol=1e-4, rtol=0)
