"""A single-response ``bf`` as the formula of the formula entry points.

brms accepts ``brm(bf(y ~ x, sigma ~ z))`` wherever it accepts a formula.
`bf` returns a `ResponseModel` for one response, which carries the response
keywords of `fit` (family, distributional and nonlinear formulas, response
roles, population coordinates). The entry points merge that value with their
own keywords here, so the single-response path has one argument contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from quivers.formulas._multivariate import ResponseModel
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas.family import Family

type PredictorFormulas = Mapping[str, str | NonlinearExpression] | None
"""Distributional or nonlinear formulas keyed by parameter name."""


class SingleResponseArguments(TypedDict):
    """The response keywords of a single-response entry point call.

    Each key has the meaning of the `fit` keyword of the same name.
    """

    formula: str
    family: str | Family
    distributional: dict[str, str | NonlinearExpression] | None
    nonlinear: dict[str, str | NonlinearExpression] | None
    fixed_parameters: dict[str, float] | None
    weights: str | None
    decision: str | None
    binomial_trials: int | str | None
    thresholds_by: str | None
    mixture_components: int
    center: bool | None
    decomp: Literal["none", "QR"]
    cmc: bool
    sparse: bool
    unused: tuple[str, ...]


def _mapping[V](values: Mapping[str, V] | None) -> dict[str, V] | None:
    """Copy an optional mapping into a plain dictionary.

    Parameters
    ----------
    values : Mapping[str, V] or None
        Authored mapping.

    Returns
    -------
    dict[str, V] or None
        A mutable copy, or None.
    """
    return None if values is None else dict(values)


def _merge[T](name: str, authored: T, keyword: T, default: T, caller: str) -> T:
    """Choose one value of a response option given in two places.

    Parameters
    ----------
    name : str
        Option name.
    authored : T
        Value carried by the ``bf`` call.
    keyword : T
        Value of the entry point's keyword.
    default : T
        The option's default, which marks a value as not given.
    caller : str
        Entry point named in the error.

    Returns
    -------
    T
        The value that was given, or the default.

    Raises
    ------
    ValueError
        If both places give different values.
    """
    if keyword is default or keyword == default:
        return authored
    if authored is default or authored == default or authored == keyword:
        return keyword
    raise ValueError(
        f"{caller}: {name} is given both in bf() ({authored!r}) and as a "
        f"keyword ({keyword!r}); give it once"
    )


def single_response(
    model: ResponseModel,
    *,
    family: str | Family,
    distributional: PredictorFormulas,
    nonlinear: PredictorFormulas,
    fixed_parameters: Mapping[str, float] | None,
    weights: str | None,
    decision: str | None,
    binomial_trials: int | str | None,
    thresholds_by: str | None,
    mixture_components: int,
    center: bool | None,
    decomp: Literal["none", "QR"],
    cmc: bool,
    sparse: bool,
    unused: Sequence[str],
    caller: str,
) -> SingleResponseArguments:
    """Merge a single-response ``bf`` with an entry point's response keywords.

    Parameters
    ----------
    model : ResponseModel
        The value `bf` returned for one response.
    family : str or Family
        The entry point's ``family`` keyword.
    distributional, nonlinear : Mapping or None
        The entry point's predictor-formula keywords.
    fixed_parameters : Mapping[str, float] or None
        The entry point's fixed auxiliary parameters.
    weights, decision, thresholds_by : str or None
        The entry point's response-role keywords.
    binomial_trials : int or str or None
        The entry point's trial count or column.
    mixture_components : int
        The entry point's mixture component count.
    center : bool or None
        The entry point's ``center`` keyword.
    decomp : {"none", "QR"}
        The entry point's ``decomp`` keyword.
    cmc, sparse : bool
        The entry point's ``cmc`` and ``sparse`` keywords.
    unused : Sequence[str]
        The entry point's ``unused`` keyword.
    caller : str
        Entry point named in errors.

    Returns
    -------
    SingleResponseArguments
        The formula string and every response option, each the value given
        in ``bf`` or as a keyword.

    Raises
    ------
    ValueError
        If an option is given in both places with different values, or a
        parameter formula of a ``bf`` without a family names no parameter
        of the entry point's family.

    Notes
    -----
    Cost is constant in the data.
    """
    # A bf without a family takes the entry point's, as brm(bf(...), family=).
    if not model.family_given:
        model = model.with_family(family)
    return SingleResponseArguments(
        formula=model.formula,
        family=_merge("family", model.family, family, "gaussian", caller),
        distributional=_merge(
            "distributional",
            _mapping(model.distributional),
            _mapping(distributional),
            None,
            caller,
        ),
        nonlinear=_merge(
            "nonlinear",
            _mapping(model.nonlinear),
            _mapping(nonlinear),
            None,
            caller,
        ),
        fixed_parameters=_merge(
            "fixed_parameters",
            _mapping(model.fixed_parameters),
            _mapping(fixed_parameters),
            None,
            caller,
        ),
        weights=_merge("weights", model.weights, weights, None, caller),
        decision=_merge("decision", model.decision, decision, None, caller),
        binomial_trials=_merge(
            "binomial_trials", model.binomial_trials, binomial_trials, None, caller
        ),
        thresholds_by=_merge(
            "thresholds_by", model.thresholds_by, thresholds_by, None, caller
        ),
        mixture_components=_merge(
            "mixture_components",
            model.mixture_components,
            mixture_components,
            2,
            caller,
        ),
        center=_merge("center", model.center, center, None, caller),
        decomp=_merge("decomp", model.decomp, decomp, "none", caller),
        cmc=_merge("cmc", model.cmc, cmc, True, caller),
        sparse=_merge("sparse", model.sparse, sparse, False, caller),
        unused=_merge("unused", tuple(model.unused), tuple(unused), (), caller),
    )


__all__: list[str] = []
