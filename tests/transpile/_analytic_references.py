"""Independent density formulas for references with cancelling float32 terms.

These references are evaluated from the fixed point, without compiling QVR or
reading a trace. A stored float32 joint is not portable when positive and
negative terms nearly cancel: roundoff is set by the terms, not their sum.
The analytic reference retains the existing whole-joint comparison budget.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import distributions as td

from tests.transpile import _gallery_data
from tests.transpile.probes._protocol import Point


def kumaraswamy_terms(
    dataset: _gallery_data.GalleryDataset,
    point: Point,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Evaluate the regression's three priors and bounded likelihood.

    Parameters
    ----------
    dataset
        Observed values and their shapes; no compiled program is read.
    point
        The clamped coefficients, global shape, covariate and response.
    dtype
        Arithmetic precision for the independent calculation.

    Returns
    -------
    dict[str, torch.Tensor]
        One summed density term per stochastic site.
    """
    b, g0, g1 = (
        torch.as_tensor(point.params[name], dtype=dtype)
        for name in ("b", "gamma_0", "gamma_1")
    )
    observations = _gallery_data.observations_for_point(dataset, point)
    x, y = (observations[name].to(dtype) for name in ("x", "y"))
    a = (g0 + g1 * x).exp()
    zero, one, two = (torch.tensor(value, dtype=dtype) for value in (0.0, 1.0, 2.0))
    return {
        "b": td.HalfNormal(two).log_prob(b).sum(),
        "gamma_0": td.Normal(zero, one).log_prob(g0).sum(),
        "gamma_1": td.Normal(zero, one).log_prob(g1).sum(),
        "y": td.Kumaraswamy(a, b).log_prob(y).sum(),
    }


ANALYTIC_REFERENCES: dict[
    str, Callable[[_gallery_data.GalleryDataset, Point], dict[str, torch.Tensor]]
] = {"kumaraswamy_bounded_outcome": kumaraswamy_terms}
