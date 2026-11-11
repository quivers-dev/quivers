"""Posterior diagnostics: convergence estimators and the ArviZ adapter.

Quivers' inference layer produces [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult]
records and [`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide] fits.
This package holds two kinds of analysis over them.

The convergence estimators (`rhat`, `ess`, `mcse`, `summary`) take any
``(num_chains, num_draws, *event_shape)`` tensor and need only torch.
`MCMCResult` reports its diagnostics through the same functions, so
derived quantities and draws from custom sampling loops are diagnosed
with the estimators a run reports.

The simulation-based calibration tools (`sbc`, `sbc_ranks`,
`sbc_uniformity`, `sbc_coverage`) check an inference procedure against
the program it targets; they also need only torch, and are bound on
first access because they build on the inference layer.

The remaining functions convert quivers fits into the
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html)
data model of [ArviZ](https://python.arviz.org/) and wrap the
canonical ArviZ entry points (`arviz.compare`, `arviz.loo`,
`arviz.plot_ppc`) with shape-aware, quivers-typed signatures; ArviZ
owns those analytics. They need the ``diagnostics`` extra and are
bound on first access, which also keeps this package importable by
the MCMC driver that the adapter itself imports.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
from types import FunctionType, GenericAlias, UnionType
from typing import TYPE_CHECKING

from quivers.diagnostics.convergence import (
    ConvergenceSummary,
    EssMethod,
    McseMethod,
    ess,
    mcse,
    rhat,
    summary,
)

if TYPE_CHECKING:
    import numpy as np

    from quivers.diagnostics.arviz_io import (
        to_datatree,
        to_datatree_any,
        to_datatree_from_sbc,
        to_datatree_from_svi,
    )
    from quivers.diagnostics.comparison import compare, loo
    from quivers.diagnostics.predictive_checks import (
        STATISTICS,
        loo_pit,
        posterior_predictive_check,
    )
    from quivers.diagnostics.calibration import (
        SBCCoverage,
        SBCPosterior,
        SBCResult,
        SBCStatistic,
        SBCThinning,
        sbc,
        sbc_coverage,
        sbc_ranks,
        sbc_uniformity,
    )

_LAZY_EXPORTS: dict[str, str] = {
    "sbc": "quivers.diagnostics.calibration",
    "sbc_ranks": "quivers.diagnostics.calibration",
    "sbc_uniformity": "quivers.diagnostics.calibration",
    "sbc_coverage": "quivers.diagnostics.calibration",
    "SBCResult": "quivers.diagnostics.calibration",
    "SBCCoverage": "quivers.diagnostics.calibration",
    "SBCStatistic": "quivers.diagnostics.calibration",
    "SBCPosterior": "quivers.diagnostics.calibration",
    "SBCThinning": "quivers.diagnostics.calibration",
    "to_datatree": "quivers.diagnostics.arviz_io",
    "to_datatree_from_sbc": "quivers.diagnostics.arviz_io",
    "to_datatree_any": "quivers.diagnostics.arviz_io",
    "to_datatree_from_svi": "quivers.diagnostics.arviz_io",
    "compare": "quivers.diagnostics.comparison",
    "loo": "quivers.diagnostics.comparison",
    "posterior_predictive_check": "quivers.diagnostics.predictive_checks",
    "loo_pit": "quivers.diagnostics.predictive_checks",
    "STATISTICS": "quivers.diagnostics.predictive_checks",
}

type _LazyExport = (
    FunctionType
    | type
    | UnionType
    | GenericAlias
    | dict[str, Callable[[np.ndarray], float]]
)


def __getattr__(name: str) -> _LazyExport:
    """Bind an SBC or ArviZ-backed export on first access.

    Parameters
    ----------
    name : str
        The attribute requested.

    Returns
    -------
    _LazyExport
        The function, class, type alias, or statistic table the
        defining module binds under that name.

    Raises
    ------
    AttributeError
        If ``name`` is not a lazily bound export.
    ModuleNotFoundError
        If the ``diagnostics`` extra (ArviZ, xarray) is not installed.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: _LazyExport = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "rhat",
    "ess",
    "mcse",
    "summary",
    "ConvergenceSummary",
    "EssMethod",
    "McseMethod",
    "sbc",
    "sbc_ranks",
    "sbc_uniformity",
    "sbc_coverage",
    "SBCResult",
    "SBCCoverage",
    "SBCStatistic",
    "SBCPosterior",
    "SBCThinning",
    "to_datatree",
    "to_datatree_from_sbc",
    "to_datatree_any",
    "to_datatree_from_svi",
    "compare",
    "loo",
    "posterior_predictive_check",
    "loo_pit",
    "STATISTICS",
]
