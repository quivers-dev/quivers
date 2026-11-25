"""Posterior contribution of one fitted smooth term at explicit rows."""

from __future__ import annotations


from collections.abc import Mapping

import narwhals as nw
import torch
from narwhals.typing import IntoDataFrame

from quivers.formulas._row_source import PredictorRows, RowSource
from quivers.formulas._gp import (
    GaussianProcess,
)
from quivers.formulas._fit_types import SingleFit
from quivers.formulas._smooth import SmoothTransform


def conditional_smooths(
    fitted: SingleFit,
    term: str,
    new_data: IntoDataFrame,
    *,
    parameter: str = "mu",
    num_samples: int | None = None,
    seed: int = 0,
    draws: Mapping[str, torch.Tensor] | None = None,
    site_prefix: str = "",
) -> torch.Tensor:
    """Evaluate one smooth's posterior contribution to its linear predictor.

    The contribution is the smooth's unpenalized columns times their
    population coefficients plus each penalized block times its scale and
    innovations, evaluated through the retained training basis, as brms's
    ``conditional_smooths`` reports. For an approximate ``gp()`` term it is
    the retained eigenfunction basis times ``sqrt(spd) * zgp`` of every
    sub-GP, which brms reports through ``conditional_effects``. For an exact
    ``gp()`` term each draw's latent function is drawn at the rows' distinct
    inputs given its training values, as brms's ``.predictor_gp_new`` does,
    at O(u^3 + u^2 m + u m^2 + m^3) per draw for ``u`` training and ``m``
    fresh inputs. Other terms, the intercept, and the
    inverse link are excluded. Costs O(draws times rows times basis width).

    Parameters
    ----------
    fitted : BayesianFit
        Fitted formula model.
    term : str
        Smooth or GP term exactly as the fitted formula's evaluator names
        it, for instance ``"s(x)"``, ``"t2(x, z, k=4)"``, or
        ``"gp(x, k=10)"``.
    new_data : IntoDataFrame
        Rows holding the smooth's covariates and any ``by`` variable.
    parameter : str
        ``mu`` or a distributional or nonlinear parameter carrying the term.
    num_samples : int or None
        Draw count under the convention of ``draws``.
    seed : int
        Seed for variational draws and for the latent function of an exact
        ``gp()`` term at the rows.
    draws : Mapping[str, torch.Tensor] or None
        Constrained draws to read in place of the fit's own: the draws of a
        joint fit when ``fitted`` is the view of one of its responses.
    site_prefix : str
        Namespace of that response's sites within ``draws``.

    Returns
    -------
    torch.Tensor
        Contributions with shape ``(chain, draw, rows)``, followed by the free
        coordinates of a vector-valued parameter.

    Raises
    ------
    ValueError
        If the fit is joint, the term is not a fitted smooth of the
        predictor, or the rows are invalid for its retained basis.
    """
    if hasattr(fitted, "response_fits"):
        raise ValueError(
            "conditional_smooths evaluates the smooth of one response; for a "
            "joint fit call its conditional_smooths method with resp"
        )
    formula, prefix = fitted._formula_predictor(parameter)
    plan = fitted.prediction_plan
    if plan is None or parameter not in plan.designs:
        raise ValueError("this fit has no retained design for smooth evaluation")
    design = plan.designs[parameter]
    terms = design.common.terms if design.common is not None else {}
    if term not in terms:
        raise ValueError(
            f"{term!r} is not a fitted term of {parameter!r}; terms: {sorted(terms)}"
        )
    components = getattr(terms[term], "components", ())
    transform = (
        getattr(getattr(components[0], "call", None), "stateful_transform", None)
        if len(components) == 1
        else None
    )
    if (
        not isinstance(transform, (SmoothTransform, GaussianProcess))
        or transform.state is None
    ):
        raise ValueError(f"{term!r} is not a fitted s(), t2(), or gp() term")
    frame = nw.from_native(new_data, eager_only=True).to_pandas()
    draws = dict(
        fitted.draws(num_samples=num_samples, seed=seed) if draws is None else draws
    )
    if site_prefix:
        draws = {
            site.removeprefix(site_prefix): value
            for site, value in draws.items()
            if site.startswith(site_prefix)
        }
    predictor = next(item for item in plan.predictors if item.prefix == prefix)
    rows = PredictorRows(RowSource(frame=frame), design)
    generator = torch.Generator().manual_seed(seed)
    parts = [
        part.at(rows).effect(draws, generator)
        for item in predictor.terms
        if (part := item.smooth_part(term)) is not None
    ]
    if not parts:
        raise ValueError(f"{term!r} has no share in the predictor of {parameter!r}")
    return sum(parts[1:], start=parts[0])


__all__ = ["conditional_smooths"]
