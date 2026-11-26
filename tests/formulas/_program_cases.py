"""Programs, data and helpers shared by the tests of program formulas.

The programs are those of the formula goldens: the approximate-numeral model
of the issue (`AROUND`, one reading shared by every row), its grouped form
(`AROUND_ROWS`, one reading per row), a Gaussian (`GAUSS`), a two-class
mixture of Gaussians (`MIX2`), an exponential decay (`DECAY`), a bounded lapse
rate (`BOUNDED`) and a simplex of class weights (`SIMPLEX`).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch

from quivers.effects.observation_scoring import observation_scoring
from quivers.formulas import BayesianFit
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import Trace, trace

AROUND = """object Trial : FinSet 6
object Res : FinSet 3
object Val : Real 1

program around : Trial -> Val
    sample res_logits <- Normal(loc=0.0, scale=1.0) [over=Res]
    sample noise <- Beta(concentration1=1.0, concentration0=20.0)
    let res_probs = softmax(res_logits)
    marginalize r : Res <- Categorical(probs=res_probs) [reduction=logsumexp]
        let inside = real(dist <= halfwidth[r])
        let p = noise + (1.0 - 2.0 * noise) * inside
        observe accept : Trial <- Bernoulli(probs=p)
    return noise

export around
"""

AROUND_ROWS = """object Trial : FinSet 6
object Res : FinSet 3
object Val : Real 1

program around_rows(trial_idx : Tensor[Int]([Trial])) : Trial -> Val
    sample res_logits <- Normal(loc=0.0, scale=1.0) [over=Res]
    sample noise <- Beta(concentration1=1.0, concentration0=20.0)
    let res_probs = softmax(res_logits)
    marginalize r : Res <- Categorical(probs=res_probs) [over=Trial, reduction=logsumexp]
        let inside = real(dist <= halfwidth[r])
        let p = noise + (1.0 - 2.0 * noise) * inside
        observe accept : Trial <- Bernoulli(probs=p) [via=trial_idx]
    return noise

export around_rows
"""

GAUSS = """object Trial : FinSet 6
object Val : Real 1

program gauss : Trial -> Val
    sample mu <- Normal(loc=0.0, scale=5.0)
    sample sigma <- HalfNormal(scale=1.0)
    observe y : Trial <- Normal(loc=mu, scale=sigma)
    return mu

export gauss
"""

MIX2 = """object Trial : FinSet 6
object Cls : FinSet 2
object Val : Real 1

program mix2(trial_idx : Tensor[Int]([Trial])) : Trial -> Val
    sample logits <- Normal(loc=0.0, scale=1.0) [over=Cls]
    sample centers <- Normal(loc=0.0, scale=2.0) [over=Cls]
    sample sigma <- HalfNormal(scale=1.0)
    let probs = softmax(logits)
    marginalize k : Cls <- Categorical(probs=probs) [over=Trial, reduction=logsumexp]
        let m = centers[k]
        observe y : Trial <- Normal(loc=m, scale=sigma) [via=trial_idx]
    return sigma

export mix2
"""

DECAY = """object Trial : FinSet 6
object Val : Real 1

program decay : Trial -> Val
    sample a <- Normal(loc=0.0, scale=2.0)
    sample b <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    let mu = a * exp(0.0 - b * x)
    observe y : Trial <- Normal(loc=mu, scale=sigma)
    return a

export decay
"""

BOUNDED = """object Trial : FinSet 6
object Val : Real 1

program bounded : Trial -> Val
    sample lapse <- Uniform(low=0.0, high=0.5)
    sample theta <- Normal(loc=0.0, scale=1.0)
    let p = lapse + (1.0 - 2.0 * lapse) * sigmoid(theta)
    observe accept : Trial <- Bernoulli(probs=p)
    return lapse

export bounded
"""

SIMPLEX = """object Trial : FinSet 6
object Res : FinSet 3
object Val : Real 1

program simplex(trial_idx : Tensor[Int]([Trial])) : Trial -> Val
    sample weights <- Dirichlet(concentration=[1.0, 1.0, 1.0]) [over=Res]
    sample sigma <- HalfNormal(scale=1.0)
    marginalize r : Res <- Categorical(probs=weights) [over=Trial, reduction=logsumexp]
        let m = halfwidth[r]
        observe y : Trial <- Normal(loc=m, scale=sigma) [via=trial_idx]
    return sigma

export simplex
"""

HALFWIDTH = np.array([1.0, 2.0, 3.0])
"""The half widths of the three readings of `AROUND`, given in ``data2``."""


def frame(rows: int = 12, seed: int = 0) -> pd.DataFrame:
    """Return rows of every column the programs read.

    Parameters
    ----------
    rows : int
        Number of rows.
    seed : int
        Seed of the values.

    Returns
    -------
    pandas.DataFrame
        Columns ``dist``, ``accept`` (Boolean), ``y``, ``x``, ``z``, the
        grouping factors ``g`` and ``h``, and the identity row map
        ``trial_idx``.
    """
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "dist": np.round(generator.uniform(0.0, 4.0, rows), 2),
            "accept": generator.uniform(size=rows) < 0.5,
            "y": np.round(generator.normal(1.0, 1.5, rows), 3),
            "x": np.round(generator.uniform(-1.0, 1.0, rows), 3),
            "z": np.round(generator.uniform(-1.0, 1.0, rows), 3),
            "g": np.array(["a", "b", "c"])[np.arange(rows) % 3],
            "h": np.array(["u", "v"])[np.arange(rows) % 2],
            "trial_idx": np.arange(rows),
        }
    )


def latent_names(fitted: BayesianFit) -> tuple[str, ...]:
    """Return the latent sites of a fit.

    Parameters
    ----------
    fitted : BayesianFit
        A fit.

    Returns
    -------
    tuple[str, ...]
        The site names.
    """
    return LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names


def pin(fitted: BayesianFit, values: Mapping[str, np.ndarray]) -> BayesianFit:
    """Replace the posterior of a fit by explicit draws on one chain.

    Parameters
    ----------
    fitted : BayesianFit
        A fit.
    values : Mapping[str, numpy.ndarray]
        Draws of every latent site, the draw axis first.

    Returns
    -------
    BayesianFit
        The fit with those draws.
    """
    assert set(values) == set(latent_names(fitted)), sorted(latent_names(fitted))
    samples = {
        name: torch.as_tensor(np.asarray(value), dtype=torch.float64).unsqueeze(0)
        for name, value in values.items()
    }
    count = next(iter(samples.values())).shape[1]
    return fitted.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, count),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def run(fitted: BayesianFit, latents: Mapping[str, torch.Tensor]) -> Trace:
    """Run a fit's program once at given latent values.

    Parameters
    ----------
    fitted : BayesianFit
        A fit.
    latents : Mapping[str, torch.Tensor]
        A value of every latent site.

    Returns
    -------
    Trace
        The trace, with the observations scored.
    """
    with torch.no_grad(), observation_scoring(True):
        return trace(
            fitted.program,
            torch.zeros(1, 1, dtype=torch.long),
            dict(fitted.observations) | dict(latents),
        )
