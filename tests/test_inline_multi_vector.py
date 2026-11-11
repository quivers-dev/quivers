"""Inline distributions with more than one vector-typed parameter.

``MixedInlineDistribution`` stacks a distribution's variable parameters
into a single input tensor and splits them back out by declared dim, so
a family may declare any number of vector-typed parameters. The
canonical case is ``MixtureNormal(weights, locations, scales)``, whose
three per-component vectors are each shared across the response plate
and define a finite Gaussian mixture scored per row.

Run with::

    pytest tests/test_inline_multi_vector.py
"""

from __future__ import annotations

import textwrap
from typing import cast

import torch
import torch.nn as nn

from quivers.continuous.inline import make_inline_distribution
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.dsl.ast_nodes import DrawArgName, DrawArgNamed


def _mixture_args(weights: str, loc: str, scale: str) -> tuple[DrawArgNamed, ...]:
    return tuple(
        DrawArgNamed(parameter=parameter, value=DrawArgName(text=value))
        for parameter, value in (
            ("weights", weights),
            ("loc", loc),
            ("scale", scale),
        )
    )


def _reference_mixture_logprob(
    weights: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Closed-form per-row log-density of a finite Gaussian mixture."""
    per_component = (
        torch.distributions.Normal(loc, scale).log_prob(y[:, None]) + weights.log()
    )
    return torch.logsumexp(per_component, dim=-1)


def test_inline_mixture_normal_log_prob_matches_reference() -> None:
    """Three shared per-component vectors (weights, loc, scale) split
    cleanly and score each row against the closed-form mixture."""
    k, n = 3, 64
    types = {
        "w": Euclidean(name="w", dim=k),
        "m": Euclidean(name="m", dim=k),
        "s": Euclidean(name="s", dim=k),
    }
    morph, order = make_inline_distribution(
        "MixtureNormal",
        _mixture_args("w", "m", "s"),
        Euclidean(name="R", dim=1),
        types,
    )
    assert order == ("w", "m", "s")

    weights = torch.tensor([0.3, 0.4, 0.3])
    loc = torch.tensor([-3.0, 0.0, 3.0])
    scale = torch.tensor([0.5, 0.7, 0.4])
    stacked = torch.cat([weights, loc, scale]).reshape(1, -1)
    y = torch.randn(n)

    got = morph.log_prob(stacked, y).reshape(-1)
    expected = _reference_mixture_logprob(weights, loc, scale, y)
    assert got.shape == (n,)
    assert torch.allclose(got, expected, atol=1e-5)


def test_inline_mixture_normal_per_row_gmm_compiles_and_fits() -> None:
    """The per-row Gaussian mixture program compiles and an SVI fit
    drives the loss down and recovers the component means."""
    from quivers.inference import ELBO, SVI, AutoNormalGuide

    source = textwrap.dedent(
        """
        composition log_prob [level=algebra]

        object Component : FinSet 3
        object Resp : FinSet 300

        program gmm : Resp -> Resp
            sample probs <- Dirichlet(concentration=1.0) [over=Component]
            sample mu : Component <- Normal(loc=0.0, scale=5.0)
            sample sigma : Component <- HalfNormal(scale=1.0)
            observe r : Resp <- MixtureNormal(weights=probs, loc=mu, scale=sigma)
            return probs

        export gmm
        """
    )
    model = cast(nn.Module, loads(source).morphism)
    assert model is not None

    torch.manual_seed(0)
    true_probs = torch.tensor([0.3, 0.4, 0.3])
    true_mu = torch.tensor([-4.0, 0.0, 4.0])
    true_sigma = torch.tensor([0.4, 0.5, 0.4])
    comps = torch.distributions.Categorical(true_probs).sample(torch.Size((300,)))
    r = torch.distributions.Normal(true_mu[comps], true_sigma[comps]).sample()
    x = torch.zeros(1, 1)
    obs = {"r": r, "probs": true_probs}

    guide = AutoNormalGuide(model, observed_names={"r", "probs"})
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=3e-2
    )
    svi = SVI(model, guide, optim, ELBO(num_particles=4))
    losses = [svi.step(x, obs) for _ in range(300)]
    assert losses[-1] < losses[0]


def test_inline_multi_vector_splits_by_declared_dim() -> None:
    """Distinct vector dims split at the right offsets: a length-2 and a
    length-3 shared vector recover their own slices."""
    types = {
        "a": Euclidean(name="a", dim=2),
        "b": Euclidean(name="b", dim=3),
    }
    morph, _ = make_inline_distribution(
        "MixtureNormal",
        _mixture_args("a", "a", "b"),
        Euclidean(name="R", dim=1),
        types,
    )
    # Two length-2 vectors then a length-3 vector: 7 columns total.
    stacked = torch.arange(7.0).reshape(1, -1)
    resolved = morph._resolve_params(stacked)
    assert [tuple(t.shape) for t in resolved] == [(1, 2), (1, 2), (1, 3)]
    assert torch.equal(resolved[0].reshape(-1), torch.tensor([0.0, 1.0]))
    assert torch.equal(resolved[1].reshape(-1), torch.tensor([2.0, 3.0]))
    assert torch.equal(resolved[2].reshape(-1), torch.tensor([4.0, 5.0, 6.0]))
