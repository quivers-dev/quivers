"""brms ``inits``: the initial values each chain of a formula fit started from.

The MCMC driver derives every chain's initial point from the run seed, the
chain index, and the initialization settings, before the chain draws
anything else from its stream. The initial values of a finished fit are thus
a function of the fit's retained arguments and data. `chain_initial_values`
calls `fit` again with those arguments, as a refit or a parallel chain
worker does, and stops it at the driver, which reports the initial values
and runs no chain.

Cost: one preparation and compilation of the model, and no sampling.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar

from narwhals.typing import IntoDataFrame
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.mcmc.driver import MCMC
from quivers.modeling.refit import RefitRecipe

_REQUESTED: ContextVar[bool] = ContextVar("quivers_initial_values", default=False)


class InitialValuesRead(Exception):
    """Carry the chains' initial values out of `fit`.

    Parameters
    ----------
    values : tuple[dict[str, torch.Tensor], ...]
        Constrained initial values of each chain, by site.
    """

    def __init__(self, values: tuple[dict[str, torch.Tensor], ...]) -> None:
        super().__init__("the initial values were read")
        self.values = values


def report_initial_values(
    mcmc: MCMC,
    program: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
) -> None:
    """Stop a fit at its driver when the initial values were requested.

    Parameters
    ----------
    mcmc : MCMC
        The configured driver.
    program : MonadicProgram
        Compiled formula program.
    x : torch.Tensor
        One-row input.
    observations : dict[str, torch.Tensor]
        Data by observed site.

    Raises
    ------
    InitialValuesRead
        When `chain_initial_values` is running the fit.
    """
    if _REQUESTED.get():
        raise InitialValuesRead(mcmc.chain_start_values(program, x, observations))


def chain_initial_values[F](
    recipe: RefitRecipe[IntoDataFrame, Mapping[str, object], F],
) -> list[dict[str, torch.Tensor]]:
    """Return the initial values of every chain of a fitted formula model.

    Parameters
    ----------
    recipe : RefitRecipe
        The fit's retained data and canonical `fit` arguments.

    Returns
    -------
    list[dict[str, torch.Tensor]]
        One mapping per chain, in chain order, from latent site to its
        initial value on the constrained scale: the values given through
        ``init`` or ``initial_values`` and the generated values elsewhere.

    Raises
    ------
    ValueError
        If the fit's method runs no MCMC chains.
    """
    token = _REQUESTED.set(True)
    try:
        # A saved fit is not reloaded: the arguments are run to the driver.
        recipe.run(None, method={**recipe.method, "file": None, "cores": 1})
    except InitialValuesRead as read:
        return [dict(values) for values in read.values]
    finally:
        _REQUESTED.reset(token)
    raise ValueError(
        f"method {recipe.method.get('method')!r} runs no MCMC chains, so the "
        "fit has no chain initial values"
    )


__all__: list[str] = []
