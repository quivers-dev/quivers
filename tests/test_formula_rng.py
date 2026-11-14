"""Formula inference owns its seeded stream and restores the caller's state."""

from typing import Literal

import pandas as pd
import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.formulas import BayesianFit, fit
from quivers.inference import AutoNormalGuide, MCMCResult


def _fit(method: Literal["svi", "nuts", "hmc"], seed: int) -> BayesianFit:
    """Fit one small model to exercise stream ownership, not convergence.

    Parameters
    ----------
    method
        Inference algorithm.
    seed
        Seed passed to the formula frontend.

    Returns
    -------
    BayesianFit
        Fitted guide or retained MCMC draws.
    """
    data = pd.DataFrame({"y": [0.1, 0.2, -0.1, 0.0]})
    if method == "svi":
        return fit("y ~ 1", data=data, method=method, num_steps=3, seed=seed)
    return fit(
        "y ~ 1",
        data=data,
        method=method,
        num_warmup=3,
        num_samples=3,
        num_chains=1,
        seed=seed,
    )


def _values(result: BayesianFit) -> dict[str, torch.Tensor]:
    """Read fitted values without consuming a random stream.

    Parameters
    ----------
    result
        Formula fit to inspect.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached posterior draws or variational parameters.
    """
    posterior = result.posterior
    if isinstance(posterior, MCMCResult):
        return {
            name: value.detach().clone() for name, value in posterior.samples.items()
        }
    return {
        name: value.detach().clone() for name, value in posterior.named_parameters()
    }


@pytest.mark.parametrize("method", ("svi", "nuts", "hmc"))
def test_formula_fit_reproduces_without_changing_caller_stream(
    method: Literal["svi", "nuts", "hmc"],
) -> None:
    """The fit seed determines results independently of the caller seed.

    Parameters
    ----------
    method
        Inference algorithm whose full formula path is exercised.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator().manual_seed(101).get_state())
        before = torch.random.get_rng_state().clone()
        first = _values(_fit(method, 19))
        assert torch.equal(before, torch.random.get_rng_state())
        torch.random.set_rng_state(torch.Generator().manual_seed(202).get_state())
        before = torch.random.get_rng_state().clone()
        second = _values(_fit(method, 19))
        assert torch.equal(before, torch.random.get_rng_state())
        other = _values(_fit(method, 20))
        assert torch.equal(before, torch.random.get_rng_state())
    assert first.keys() == second.keys() == other.keys()
    assert first
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    assert any(not torch.equal(first[name], other[name]) for name in first)


class _FailingGuide(AutoNormalGuide):
    """A guide that consumes randomness before raising."""

    def __init__(self, model: MonadicProgram, *, observed_names: set[str]) -> None:
        """Raise after guide initialization and a draw from the ambient stream.

        Parameters
        ----------
        model
            Compiled formula program.
        observed_names
            Sites observed by the formula fit.

        Raises
        ------
        RuntimeError
            Always, after consuming random numbers.
        """
        super().__init__(model, observed_names=observed_names)
        torch.rand(3)
        raise RuntimeError("guide initialization failed")


def test_formula_fit_restores_caller_stream_after_failure() -> None:
    """Restoration holds when inference raises after consuming randomness."""
    data = pd.DataFrame({"y": [0.1, 0.2, -0.1, 0.0]})
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="guide initialization failed"):
        fit("y ~ 1", data=data, method="svi", num_steps=1, seed=19, guide=_FailingGuide)
    assert torch.equal(before, torch.random.get_rng_state())
