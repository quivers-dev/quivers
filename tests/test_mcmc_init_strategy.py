"""Each `MCMC` initial-position strategy does what its name says.

``"prior"`` draws every latent from its prior (given the data), so its
initial positions, mapped back to the constrained space, follow the
prior: tested with Kolmogorov-Smirnov against the prior's CDF for a
real latent and a positive one. ``"jitter"`` stays near the origin of
the unconstrained space and is rejected by the same test, which shows
the test can tell the two apart.
"""

from __future__ import annotations

import pytest
import torch
from scipy import stats

from quivers.dsl import loads
from quivers.inference import MCMC, HMCKernel

_MODEL = """
object Obs : FinSet 5
program p : Obs -> Obs
    sample mu <- Normal(loc=2.0, scale=3.0)
    sample sigma <- HalfNormal(scale=1.5)
    observe y : Obs <- Normal(loc=mu, scale=sigma)
    return mu
export p
"""

# 2000 draws: a correct sampler fails a KS test at level 1e-4 with
# probability 1e-4; a sampler concentrated near the origin (jitter) has
# a KS distance far above the 0.05 the level allows at this size.
_DRAWS = 2000
_LEVEL = 1e-4


def _initial_draws(strategy: str) -> dict[str, torch.Tensor]:
    """Constrained initial positions of many chains.

    Parameters
    ----------
    strategy : str
        The init strategy.

    Returns
    -------
    dict[str, torch.Tensor]
        Each latent's initial values, one per chain.
    """
    model = loads(_MODEL).morphism
    y = torch.tensor([1.0, 2.5, 0.3, 1.9, 2.2])
    driver = MCMC(
        kernel=HMCKernel(step_size=0.1, num_steps=5),
        num_warmup=0,
        num_samples=1,
        num_chains=1,
        init_strategy=strategy,  # type: ignore[arg-type]
    )
    setup = driver._setup(model, torch.zeros(1, 1), {"y": y}, "test", run_seed=0)
    registry = setup.registry
    mus, sigmas = [], []
    for chain in range(_DRAWS):
        generator = torch.Generator().manual_seed(10_000 + chain)
        flat = driver._initial_position(setup, None, generator)
        draw, _ = registry.to_constrained(registry.unflatten_unconstrained(flat))
        mus.append(float(draw["mu"].reshape(-1)[0]))
        sigmas.append(float(draw["sigma"].reshape(-1)[0]))
    return {"mu": torch.tensor(mus), "sigma": torch.tensor(sigmas)}


def test_prior_initialization_follows_the_prior() -> None:
    """``"prior"`` initial positions pass KS against Normal(2, 3) and HalfNormal(1.5)."""
    draws = _initial_draws("prior")
    assert stats.kstest(draws["mu"].numpy(), stats.norm(2.0, 3.0).cdf).pvalue > _LEVEL
    assert (
        stats.kstest(draws["sigma"].numpy(), stats.halfnorm(scale=1.5).cdf).pvalue
        > _LEVEL
    )


def test_jitter_initialization_is_not_the_prior() -> None:
    """``"jitter"`` stays near the unconstrained origin, and the KS check rejects it."""
    draws = _initial_draws("jitter")
    assert float(draws["mu"].abs().max()) < 1.0
    assert stats.kstest(draws["mu"].numpy(), stats.norm(2.0, 3.0).cdf).pvalue < _LEVEL


@pytest.mark.parametrize("strategy", ["prior", "jitter"])
def test_initialization_is_seeded(strategy: str) -> None:
    """The same chain generator gives the same initial position."""
    first = _initial_draws(strategy)
    assert torch.equal(first["mu"][:5], _initial_draws(strategy)["mu"][:5])
