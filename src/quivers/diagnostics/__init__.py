"""Adapter to the ArviZ Bayesian-analysis ecosystem.

Quivers' inference layer produces [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult]
records and [`quivers.inference.guides.base.Guide`][quivers.inference.guides.base.Guide] fits.  The
canonical 2026 Bayesian-analysis library is [ArviZ](https://python.arviz.org/),
whose
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html)-based
data model is the lingua franca for posterior-summary, diagnostic,
posterior-predictive, model-comparison, and calibration tooling.

The functions in this subpackage convert quivers fits into ArviZ
DataTrees and wrap the canonical ArviZ entry points
(`arviz.compare`, `arviz.loo`, `arviz.hdi`,
`arviz.plot_ppc`) with shape-aware, quivers-typed signatures.
No posterior-analysis primitives are reimplemented here; ArviZ owns
the analytics.
"""

from quivers.diagnostics.arviz_io import (
    to_datatree,
    to_datatree_any,
    to_datatree_from_svi,
)
from quivers.diagnostics.comparison import compare, loo
from quivers.diagnostics.predictive_checks import (
    STATISTICS,
    loo_pit,
    posterior_predictive_check,
)

__all__ = [
    "to_datatree",
    "to_datatree_any",
    "to_datatree_from_svi",
    "compare",
    "loo",
    "posterior_predictive_check",
    "loo_pit",
    "STATISTICS",
]
