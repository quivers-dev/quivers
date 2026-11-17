"""Posterior contribution of one fitted smooth term at explicit rows."""

from __future__ import annotations


from collections.abc import Mapping

import narwhals as nw
import numpy as np
import torch
from narwhals.typing import IntoDataFrame

from quivers.formulas._gp import (
    GaussianProcess,
    exact_contribution,
    gp_block_coefficients,
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
        Contributions with shape ``(chain, draw, rows)``.

    Raises
    ------
    ValueError
        If the fit is joint, the term is not a fitted scalar-predictor
        smooth, or the rows are invalid for its retained basis.
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
    values = np.asarray(terms[term].eval_new_data(frame), dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != len(frame) or not np.isfinite(values).all():
        raise ValueError("smooth evaluation requires finite values for every row")
    draws = dict(
        fitted.draws(num_samples=num_samples, seed=seed) if draws is None else draws
    )
    prefix = site_prefix + prefix
    matrix = torch.as_tensor(values)
    total = None
    fixed = [column for column in formula.fixed_columns if column.term == term]
    for position, column in enumerate(fixed):
        coefficient = draws[prefix + "beta_" + column.qvr_name].double()
        if coefficient.ndim != 2:
            raise ValueError("conditional_smooths supports scalar linear predictors")
        part = coefficient[..., None] * matrix[:, position]
        total = part if total is None else total + part
    generator = torch.Generator().manual_seed(seed)
    for block in formula.design_terms:
        if block.term != term:
            continue
        # gp-exact: draw the latent function at the rows' distinct inputs.
        if block.gp is not None and block.gp.exact:
            part = exact_contribution(
                draws,
                block.gp,
                prefix + block.qvr_name,
                prefix,
                values[:, list(block.column_indices)],
                generator,
            )
            total = part if total is None else total + part
            continue
        # An approximate gp() block scales its innovations by sqrt(spd).
        if block.gp is not None:
            coefficients = gp_block_coefficients(
                draws, block.gp, prefix + block.qvr_name, prefix
            )
            part = coefficients @ matrix[:, list(block.column_indices)].T
            total = part if total is None else total + part
            continue
        scale = draws[
            prefix + (block.scale_group or block.qvr_name) + "_scale"
        ].double()
        raw = draws[prefix + block.qvr_name + "_z"].double()
        if scale.ndim != 2:
            raise ValueError("conditional_smooths supports scalar linear predictors")
        coefficients = scale[..., None] * raw
        part = coefficients @ matrix[:, list(block.column_indices)].T
        total = part if total is None else total + part
    assert total is not None
    return total


__all__ = ["conditional_smooths"]
