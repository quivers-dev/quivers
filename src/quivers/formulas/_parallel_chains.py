"""brms ``cores``: a formula fit's chains in separate processes.

A worker process calls `fit` with the canonical arguments and data the
parent's refit recipe retained, which is how refits and restored artifacts
rebuild a fit. Preparation, compilation, and the driver are thus those of
the parent. When the fit reaches its MCMC driver inside a worker, the driver
runs only the chains assigned to the worker and their records leave the fit
as a `ChainsRan` signal, so the worker does no assembly or prediction work.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar

import didactic.api as dx
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.inference.mcmc.parallel import ChainStates
import quivers.formulas._fit as fit_module

_ASSIGNED: ContextVar[tuple[int, ...] | None] = ContextVar(
    "quivers_assigned_chains", default=None
)


class ChainsRan(Exception):
    """Carry a worker's chain records out of `fit`.

    Parameters
    ----------
    states : ChainStates
        The records of the assigned chains.
    """

    def __init__(self, states: ChainStates) -> None:
        super().__init__("the assigned chains ran")
        self.states = states


def assigned_chains() -> tuple[int, ...] | None:
    """Return the chains this process runs, or None outside a worker.

    Returns
    -------
    tuple[int, ...] or None
        Chain indices assigned to the current worker.
    """
    return _ASSIGNED.get()


class FormulaChainWorker(ValueModel):
    """Rebuild a formula fit from its recipe and run some of its chains.

    Parameters
    ----------
    arguments : Mapping[str, object]
        Canonical `fit` arguments other than the data, as the refit recipe
        retains them.
    data : IntoDataFrame
        The retained original rows.
    """

    arguments: Mapping[str, object] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    data: IntoDataFrame = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def __call__(self, chains: tuple[int, ...]) -> ChainStates:
        """Run `fit` up to its driver with the assigned chains.

        Parameters
        ----------
        chains : tuple[int, ...]
            Chain indices.

        Returns
        -------
        ChainStates
            Their records.

        Raises
        ------
        ValueError
            If the fit finished without reaching an MCMC driver.
        """
        token = _ASSIGNED.set(chains)
        try:
            fit_module.fit(data=self.data, **self.arguments)  # type: ignore[arg-type]
        except ChainsRan as ran:
            return ran.states
        finally:
            _ASSIGNED.reset(token)
        raise ValueError("a parallel chain worker's fit ran no MCMC driver")


__all__: list[str] = []
