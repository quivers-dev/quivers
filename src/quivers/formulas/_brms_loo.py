"""brms ``loo(fit1, fit2, ...)`` and ``waic(fit1, fit2, ...)`` over several models.

brms computes the criterion of each model passed to ``loo()`` or ``waic()``
and, with more than one model, compares them (``loo_compare``). The
functions here do the same through the stored or freshly computed estimates
of the shared comparison path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics.elpd import ELPDEstimate, compare_estimates
from quivers.diagnostics.posterior_tables import SummaryTable
from quivers.formulas._postprocess_methods import ModelInput, _estimate


class CriterionList(ValueModel):
    """The estimates of several models and their comparison, brms's ``loolist``.

    Parameters
    ----------
    estimates : dict[str, ELPDEstimate]
        One estimate per model, keyed by model name, in argument order
        (brms's ``loos``).
    diffs : SummaryTable or None
        The comparison table, best model first (brms's ``diffs``); None
        for one model or with ``compare=False``.
    """

    estimates: dict[str, ELPDEstimate] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    diffs: SummaryTable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _criterion_list(
    criterion: Literal["loo", "waic"],
    models: tuple[ModelInput, ...],
    model_names: Sequence[str] | None,
    compare: bool,
) -> CriterionList:
    """Compute one criterion for several models and compare them.

    Parameters
    ----------
    criterion : {"loo", "waic"}
        Criterion.
    models : tuple
        Fits or estimates.
    model_names : Sequence[str] or None
        Names of the models; ``fit1``, ``fit2``, ... by default.
    compare : bool
        Compare the models when there are several.

    Returns
    -------
    CriterionList
        Estimates and their comparison.

    Raises
    ------
    ValueError
        If no model is given or the names do not match the models.
    """
    if not models:
        raise ValueError(f"{criterion} requires at least one model")
    names = (
        [f"fit{index}" for index in range(1, len(models) + 1)]
        if model_names is None
        else [str(name) for name in model_names]
    )
    if len(names) != len(models) or len(set(names)) != len(names):
        raise ValueError("model_names must give one distinct name per model")
    estimates = {
        name: _estimate(model, criterion, compute=True)
        for name, model in zip(names, models, strict=True)
    }
    return CriterionList(
        estimates=estimates,
        diffs=compare_estimates(estimates) if compare and len(models) > 1 else None,
    )


def loo(
    *models: ModelInput,
    model_names: Sequence[str] | None = None,
    compare: bool = True,
) -> CriterionList:
    """Compute PSIS-LOO for one or more models and compare them, as brms ``loo``.

    Parameters
    ----------
    *models : BayesianFit or MultivariateFit or ELPDEstimate
        Fitted models, or estimates. A fit with a stored ``loo`` criterion
        (`add_criterion`) is not evaluated again.
    model_names : Sequence[str] or None
        Names of the models; ``fit1``, ``fit2``, ... by default.
    compare : bool
        Compare several models, as ``loo_compare`` does.

    Returns
    -------
    CriterionList
        One estimate per model, and their comparison table for several
        models.

    Raises
    ------
    ValueError
        If no model is given, the names do not match, or the models score
        different observations.
    """
    return _criterion_list("loo", models, model_names, compare)


def waic(
    *models: ModelInput,
    model_names: Sequence[str] | None = None,
    compare: bool = True,
) -> CriterionList:
    """Compute WAIC for one or more models and compare them, as brms ``waic``.

    Parameters
    ----------
    *models : BayesianFit or MultivariateFit or ELPDEstimate
        Fitted models, or estimates.
    model_names : Sequence[str] or None
        Names of the models; ``fit1``, ``fit2``, ... by default.
    compare : bool
        Compare several models.

    Returns
    -------
    CriterionList
        One estimate per model, and their comparison table for several
        models.

    Raises
    ------
    ValueError
        If no model is given, the names do not match, or the models score
        different observations.
    """
    return _criterion_list("waic", models, model_names, compare)


__all__: list[str] = []
