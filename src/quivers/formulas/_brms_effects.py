"""Automatic conditional effects and smooths, as brms ``conditional_effects``.

brms ``conditional_effects`` [@burkner2017brms] chooses its effects from the
model: every population-level predictor variable and every pair of variables
that appear together in one term, across all predicted parameters, with the
variables of special terms (``mo``, ``me``, ``mi``, ``s``, ``t2``, ``gp``)
crossed with each other. For each effect it builds a grid over the effect's
variables (``resolution`` equally spaced values over the observed range of a
numeric variable, the levels of a factor, the mean and the mean plus and
minus one standard deviation of a numeric second variable), holds every
other variable at a condition (the mean of a numeric variable, the
rounded median of an integer variable, the reference level of a factor, one
for trial counts, and no group level), predicts the grid, and summarizes the
draws of each row. This module reproduces those choices from the fit's
retained training data and its fitted term structure, and predicts through
the fit's own prediction methods. Results are labeled data frames with
brms's columns (``effect1__``, ``effect2__``, ``cond__``, ``estimate__``,
``se__``, ``lower__``, ``upper__``), with the draws retained for paired
contrasts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import math
import re
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import didactic.api as dx
import narwhals as nw
import numpy as np
import pandas as pd
import torch
from narwhals.typing import IntoDataFrame

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.diagnostics.posterior_tables import _statistics
from quivers.formulas._postprocess_methods import (
    JointPredictionOptions,
    SinglePredictionOptions,
)
from quivers.formulas._group_prediction import NewLevelPolicy
from quivers.formulas._postprocess_predict import PredictionMethod
from quivers.formulas._response_modifiers import ExpectationOptions
from quivers.formulas._design_variables import design_variables
from quivers.formulas._smooth_effects import conditional_smooths as smooth_contribution
from quivers.formulas._fit_types import Fit

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit

type EffectMethod = Literal[
    "epred",
    "linpred",
    "predict",
    "posterior_epred",
    "posterior_linpred",
    "posterior_predict",
]
"""Prediction method of ``conditional_effects``: short or brms names."""

type ConditionValue = str | int | float | bool | None
"""A scalar condition or grid value."""

type IntCondition = (
    Sequence[ConditionValue]
    | Mapping[str, ConditionValue]
    | Callable[[pd.Series], Sequence[ConditionValue]]
)
"""``int_conditions`` entry: values, labeled values, or a function of the data."""

#: Calls whose arguments brms treats as special-term variables, by term type.
_IDENTIFIER = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*")


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one column of a frame as a series.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame.
    name : str
        Column name.

    Returns
    -------
    pandas.Series
        The column.
    """
    return cast(pd.Series, frame[name])


def _number(value: float | int | np.floating | np.integer | pd.Series) -> float:
    """Return a pandas scalar reduction as a float.

    Parameters
    ----------
    value : float or int or numpy scalar
        Reduction result.

    Returns
    -------
    float
        The value.
    """
    return float(cast(float, value))


class EffectFrame(ValueModel):
    """One conditional effect, as an element of ``brms_conditional_effects``.

    Parameters
    ----------
    name : str
        brms's list name: the effects joined by ``:``, prefixed by
        ``<resp>.`` for a joint fit.
    effects : tuple[str, ...]
        Plotted variables in plotting order; ``cats__`` for categories.
    response : str
        Response, distributional parameter, or nonlinear parameter predicted.
    frame : pandas.DataFrame
        Grid and condition columns, ``cond__``, ``effect1__`` (and
        ``effect2__``, ``cats__``), and ``estimate__``, ``se__``,
        ``lower__``, ``upper__``.
    draws : torch.Tensor
        Prediction draws ``(chain, draw, row)``, rows in frame order.
    surface : bool
        Whether both effects are numeric and plotted as a surface.
    categorical : bool
        Whether rows are per category.
    spaghetti : pandas.DataFrame or None
        One row per draw and grid row with ``estimate__`` and ``sample__``.
    points : pandas.DataFrame or None
        Observed data assigned to conditions, with ``resp__``.
    """

    name: str
    effects: tuple[str, ...]
    response: str
    frame: pd.DataFrame = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    draws: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    surface: bool = False
    categorical: bool = False
    spaghetti: pd.DataFrame | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    points: pd.DataFrame | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class BrmsConditionalEffects(ValueModel):
    """Conditional effects of one call, as a ``brms_conditional_effects`` list.

    Parameters
    ----------
    effects : tuple[EffectFrame, ...]
        Effects in brms's order.
    smooths_only : bool
        Whether the effects are smooth contributions (``conditional_smooths``).
    """

    effects: tuple[EffectFrame, ...] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    smooths_only: bool = False

    def names(self) -> tuple[str, ...]:
        """Return the effect names in order.

        Returns
        -------
        tuple[str, ...]
            brms's list names.
        """
        return tuple(effect.name for effect in self.effects)

    def __getitem__(self, name: str) -> EffectFrame:
        """Return one effect by name.

        Parameters
        ----------
        name : str
            brms's list name.

        Returns
        -------
        EffectFrame
            The effect.

        Raises
        ------
        KeyError
            If no effect has the name.
        """
        for effect in self.effects:
            if effect.name == name:
                return effect
        raise KeyError(f"no effect {name!r}; effects are {list(self.names())}")

    def frames(self) -> dict[str, pd.DataFrame]:
        """Return every effect's data frame by name.

        Returns
        -------
        dict[str, pandas.DataFrame]
            brms's data frames.
        """
        return {effect.name: effect.frame for effect in self.effects}


# --------------------------------------------------------------------------
# Data, variables and term structure
# --------------------------------------------------------------------------


def _joint(fit: Fit) -> bool:
    """Return whether a fit is a joint multivariate fit.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.

    Returns
    -------
    bool
        True for a `MultivariateFit`.
    """
    return hasattr(fit, "response_fits")


def model_frame(fit: Fit) -> pd.DataFrame:
    """Return the fitted rows of the retained training data, as ``model.frame``.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model with a refit recipe.

    Returns
    -------
    pandas.DataFrame
        The included rows of a single-response fit, or every row of a joint
        fit, with a fresh default index.

    Raises
    ------
    ValueError
        If the fit retains no training data (a restored artifact).
    """
    recipe = fit.refit_recipe
    if recipe is None:
        raise ValueError(
            "automatic conditional effects require the retained training data; "
            "pass an explicit prediction_grid for a restored fit"
        )
    frame = nw.from_native(recipe.data, eager_only=True).to_pandas()
    if not _joint(fit):
        frame = frame.iloc[list(cast("BayesianFit", fit).included_rows)]
    return frame.reset_index(drop=True)


def _views(fit: Fit, resp: str | Sequence[str] | None) -> dict[str, BayesianFit]:
    """Return the single-response views of the selected responses.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    resp : str or Sequence[str] or None
        Joint response selection; must be None for single-response fits.

    Returns
    -------
    dict[str, BayesianFit]
        Views keyed by response name (an empty key for a single response).

    Raises
    ------
    ValueError
        If ``resp`` is given for a single-response fit.
    """
    if not _joint(fit):
        if resp is not None:
            raise ValueError("resp applies only to joint fits")
        return {"": cast("BayesianFit", fit)}
    joint = cast("MultivariateFit", fit)
    return {name: joint._response_fit(name) for name in joint._selection(resp)}


def _effect_kinds(view: BayesianFit, prefix: str) -> dict[str, str]:
    """Return brms's type of each design term of one predictor.

    Parameters
    ----------
    view : BayesianFit
        Single-response fit or response view.
    prefix : str
        Namespace of the predictor.

    Returns
    -------
    dict[str, str]
        ``"fe"``, ``"sp"``, ``"sm"`` or ``"gp"`` by design term. A design term
        several terms read, such as a smooth's unpenalized columns, has the
        type of the most specific one.
    """
    plan = view.prediction_plan
    kinds: dict[str, str] = {}
    for predictor in plan.predictors if plan is not None else ():
        if predictor.prefix != prefix:
            continue
        for term in predictor.terms:
            for source in term.effects():
                if source.kind != "fe" or source.label not in kinds:
                    kinds[source.label] = source.kind
    return kinds


def _crossed(variables: list[str]) -> list[tuple[str, ...]]:
    """Return the terms of ``~ v1 * v2 * ...`` up to order two.

    Parameters
    ----------
    variables : list[str]
        Variables in order.

    Returns
    -------
    list[tuple[str, ...]]
        Main effects, then pairs in R's ``terms`` order.
    """
    singles = [(v,) for v in variables]
    pairs = list(combinations(variables, 2))
    return singles + pairs


def _parameter_effects(view: BayesianFit) -> list[tuple[str, ...]]:
    """Return brms's ``get_all_effects`` of one response.

    Parameters
    ----------
    view : BayesianFit
        Single-response fit or response view.

    Returns
    -------
    list[tuple[str, ...]]
        Variable combinations of at most two variables, predicted parameter
        by parameter: population-level terms, then special terms by type.
    """
    plan = view.prediction_plan
    out: list[tuple[str, ...]] = []
    if plan is None:
        return out
    prefixes = {name: prefix for name, prefix, _ in view._predictor_items()}
    for parameter, design in plan.designs.items():
        if design.common is None:
            continue
        kinds = _effect_kinds(view, prefixes[parameter])
        typed: dict[str, list[list[str]]] = {"fe": [], "sp": [], "sm": [], "gp": []}
        for label, term in design.common.terms.items():
            variables = list(design_variables(term).names)
            if variables and label in kinds:
                typed[kinds[label]].append(variables)
        # R's terms() orders population-level terms by interaction order.
        ordered_terms = sorted(typed["fe"], key=len)
        out.extend(tuple(v) for v in ordered_terms)
        for kind in ("sp", "sm", "gp"):
            for variables in typed[kind]:
                out.extend(_crossed(variables))
    graph = view.predictor_graph
    if graph is not None and graph.inputs:
        # The data columns of a custom family (brms vreal()/vint()) are
        # addition-term data, which brms does not plot as effects.
        custom = view.family.custom
        reserved = {column for _, column in custom.data} if custom is not None else ()
        covariates = [name for name in graph.inputs if name not in reserved]
        out.extend((v,) for v in covariates)
        out.extend(combinations(covariates, 2))
    unique: list[tuple[str, ...]] = []
    for effect in out:
        if len(effect) <= 2 and effect not in unique:
            unique.append(effect)
    return unique


def _formula_texts(view: BayesianFit) -> list[str]:
    """Return the authored formula texts of a response.

    Parameters
    ----------
    view : BayesianFit
        Single-response fit or response view.

    Returns
    -------
    list[str]
        The main formula, then each distributional and nonlinear formula,
        then the data columns of a custom family (brms ``vreal``/``vint``).
    """
    texts = [view.formula.formula]
    texts.extend(f.formula for f in view.distributional.values())
    if view.predictor_graph is not None:
        texts.extend(f.formula for f in view.predictor_graph.leaves.values())
    if view.family.custom is not None:
        texts.extend(column for _, column in view.family.custom.data)
    return texts


def model_variables(views: Mapping[str, BayesianFit], data: pd.DataFrame) -> list[str]:
    """Return the data variables of the formulas, in order of appearance.

    Parameters
    ----------
    views : Mapping[str, BayesianFit]
        Response views.
    data : pandas.DataFrame
        Model frame.

    Returns
    -------
    list[str]
        Responses, predictors, group variables and addition-term variables.
    """
    found: list[str] = []
    for view in views.values():
        for text in _formula_texts(view):
            for match in _IDENTIFIER.finditer(text):
                token = match.group(0)
                if token in data.columns and token not in found:
                    found.append(token)
    return found


def _group_variables(views: Mapping[str, BayesianFit], data: pd.DataFrame) -> set[str]:
    """Return the grouping variables of the varying-effect terms.

    Parameters
    ----------
    views : Mapping[str, BayesianFit]
        Response views.
    data : pandas.DataFrame
        Model frame.

    Returns
    -------
    set[str]
        Data columns that name a group, including the parts of ``g1:g2``.
    """
    out: set[str] = set()
    for view in views.values():
        for term in view.formula.random_terms:
            out.update(p for p in re.split(r"[:/]", term.group) if p in data.columns)
    return out


def _integer_variables(views: Mapping[str, BayesianFit]) -> set[str]:
    """Return brms's ``get_int_vars``: trial counts and monotonic variables.

    Parameters
    ----------
    views : Mapping[str, BayesianFit]
        Response views.

    Returns
    -------
    set[str]
        Variables valued on integers.
    """
    out: set[str] = set()
    for view in views.values():
        out.update(_trial_variables(view))
        plan = view.prediction_plan
        # Every predictor: location, distributional, and per-category.
        for design in plan.designs.values() if plan is not None else ():
            if design.common is not None:
                for term in design.common.terms.values():
                    out.update(design_variables(term).integer)
    return out


def _trial_variables(view: BayesianFit) -> list[str]:
    """Return the data columns of the ``trials`` addition term.

    Parameters
    ----------
    view : BayesianFit
        Response view.

    Returns
    -------
    list[str]
        The trial-count column, if any.
    """
    return [
        str(value)
        for key, value in view.formula.modifiers.sources
        if key == "trials" and isinstance(value, str)
    ]


def _like_factor(values: pd.Series) -> bool:
    """Return brms's ``is_like_factor``: factor, character or logical data.

    Parameters
    ----------
    values : pandas.Series
        Column.

    Returns
    -------
    bool
        True for categorical, string, object or Boolean columns.
    """
    return bool(
        isinstance(values.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(values)
        or pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
    )


def _levels(values: pd.Series) -> list[ConditionValue]:
    """Return the factor levels of a column, as ``levels(factor(x))``.

    Parameters
    ----------
    values : pandas.Series
        Factor-like column.

    Returns
    -------
    list
        Categories in order for a categorical column (unused ones dropped),
        else the sorted distinct values.
    """
    present = values.dropna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        used = set(present.tolist())
        return [c for c in values.cat.categories.tolist() if c in used]
    return sorted(set(present.tolist()), key=lambda v: (str(type(v)), v))


def _r_text(value: ConditionValue) -> str:
    """Format a value as R's ``as.character``.

    Parameters
    ----------
    value : str or int or float or bool or None
        Scalar.

    Returns
    -------
    str
        Fifteen significant digits for numbers, ``TRUE``/``FALSE``, ``NA``.
    """
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number == 0.0:
            return "0"
        return format(number, ".15g")
    return str(value)


def _r_round(value: float, digits: int) -> float:
    """Round as R's ``round`` for display labels.

    Parameters
    ----------
    value : float
        Number.
    digits : int
        Decimal places.

    Returns
    -------
    float
        Rounded value.
    """
    return float(round(float(value), digits))


def rows2labels(
    frame: pd.DataFrame, *, digits: int = 2, sep: str = " & ", incl_vars: bool = True
) -> list[str]:
    """Label condition rows, as brms ``rows2labels``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Condition rows.
    digits : int
        Decimal places of numeric values.
    sep : str
        Separator between variables.
    incl_vars : bool
        Prefix each value with ``<name> = ``.

    Returns
    -------
    list[str]
        One label per row.
    """
    parts: list[list[str]] = []
    for name in frame.columns:
        column = _col(frame, name)
        numeric = not _like_factor(column)
        texts = [
            _r_text(_r_round(v, digits) if numeric and v is not None else v)
            for v in column.tolist()
        ]
        parts.append([f"{name} = {t}" if incl_vars else t for t in texts])
    return [sep.join(row) for row in zip(*parts, strict=True)]


def make_conditions(
    data: IntoDataFrame | Fit,
    variables: str | Sequence[str],
    **labels: str | int | bool,
) -> pd.DataFrame:
    """Build conditions from variables of the data, as brms ``make_conditions``.

    Parameters
    ----------
    data : IntoDataFrame or BayesianFit or MultivariateFit
        A data frame, or a fit whose retained training data is used.
    variables : str or Sequence[str]
        Variables to condition on.
    **labels : str or int or bool
        `rows2labels` options (``digits``, ``sep``, ``incl_vars``).

    Returns
    -------
    pandas.DataFrame
        Every combination of the levels of factor variables and of the mean
        and mean plus and minus one standard deviation of numeric variables,
        the last variable varying fastest, with ``cond__`` labels.

    Raises
    ------
    ValueError
        If a variable is not a column of the data.
    """
    if hasattr(data, "refit_recipe"):
        frame = model_frame(cast("Fit", data))
    else:
        frame = nw.from_native(cast(IntoDataFrame, data), eager_only=True).to_pandas()
    names = [variables] if isinstance(variables, str) else list(variables)
    missing = [v for v in names if v not in frame.columns]
    if missing:
        raise ValueError(f"make_conditions: variables {missing} are not in the data")
    values: dict[str, list[ConditionValue]] = {}
    for name in names:
        column = _col(frame, name)
        if _like_factor(column):
            values[name] = _levels(column)
        else:
            mean = _number(column.mean())
            sd = _number(column.std(ddof=1))
            values[name] = [mean - sd, mean, mean + sd]
    index = pd.MultiIndex.from_product([values[n] for n in names], names=names)
    out = index.to_frame(index=False)
    for name in names:
        if _like_factor(_col(frame, name)) and isinstance(
            _col(frame, name).dtype, pd.CategoricalDtype
        ):
            out[name] = pd.Categorical(
                out[name], categories=_col(frame, name).cat.categories
            )
    out["cond__"] = rows2labels(
        cast(pd.DataFrame, out[names]),
        digits=int(labels.get("digits", 2)),
        sep=str(labels.get("sep", " & ")),
        incl_vars=bool(labels.get("incl_vars", True)),
    )
    return out


# --------------------------------------------------------------------------
# Conditions and grids
# --------------------------------------------------------------------------


class _Context(ValueModel):
    """Model-level inputs shared by every effect of a call.

    Parameters
    ----------
    data : pandas.DataFrame
        Model frame.
    variables : tuple[str, ...]
        Model variables in order.
    group_vars : tuple[str, ...]
        Grouping variables.
    int_vars : tuple[str, ...]
        Integer-valued variables.
    """

    data: pd.DataFrame = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    variables: tuple[str, ...]
    group_vars: tuple[str, ...]
    int_vars: tuple[str, ...]


def prepare_conditions(
    context: _Context,
    views: Mapping[str, BayesianFit],
    conditions: IntoDataFrame | Mapping[str, Sequence[ConditionValue]] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Complete conditions with brms's defaults, as ``prepare_conditions``.

    Parameters
    ----------
    context : _Context
        Model frame and variable classes.
    views : Mapping[str, BayesianFit]
        Response views.
    conditions : IntoDataFrame or Mapping or None
        User conditions, one row per condition, optionally with ``cond__``.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        Completed conditions (without ``cond__``) and their labels.

    Raises
    ------
    ValueError
        If the conditions are empty or their labels repeat.
    """
    data = context.data
    required = [v for v in context.variables]
    if conditions is None:
        frame = pd.DataFrame({v: [None] for v in required})
        labels = ["1"]
    else:
        frame = (
            pd.DataFrame(dict(conditions))
            if isinstance(conditions, Mapping)
            else nw.from_native(conditions, eager_only=True).to_pandas()
        )
        if not len(frame):
            raise ValueError("conditions must have at least one row")
        frame = frame.drop_duplicates()
        if "cond__" in frame.columns:
            labels = [str(v) for v in frame["cond__"].tolist()]
            frame = frame.drop(columns="cond__")
        else:
            labels = [str(i + 1) for i in frame.index.tolist()]
        if len(set(labels)) != len(labels):
            raise ValueError("condition labels must be unique")
        frame = frame.reset_index(drop=True)
        required = [v for v in required if v not in frame.columns]
    trial_vars = [v for view in views.values() for v in _trial_variables(view)]
    for v in trial_vars:
        if v not in frame.columns or _col(frame, v).isna().all():
            frame[v] = 1
            if v in required:
                required.remove(v)
    for v in required:
        if v in context.group_vars:
            if v not in frame.columns:
                frame[v] = None
            continue
        column = _col(data, v)
        if _like_factor(column):
            value: ConditionValue = _levels(column)[0]
        elif v in context.int_vars:
            value = float(round(_number(column.median())))
        else:
            value = _number(column.mean())
        frame[v] = [value] * len(frame)
    return frame, labels


def _numeric_grid(values: pd.Series, integer: bool, resolution: int) -> list[float]:
    """Return brms's grid over a numeric variable's range.

    Parameters
    ----------
    values : pandas.Series
        Observed values.
    integer : bool
        Step by one instead of ``resolution`` equal steps.
    resolution : int
        Grid size.

    Returns
    -------
    list[float]
        Grid values from the minimum to the maximum.
    """
    low, high = float(values.min()), float(values.max())
    if integer:
        return [low + i for i in range(int(math.floor(high - low)) + 1)]
    if resolution == 1:
        return [low]
    step = (high - low) / (resolution - 1)
    return [low + i * step for i in range(resolution)]


def _int_condition(
    spec: IntCondition, values: pd.Series
) -> tuple[list[ConditionValue], list[str] | None]:
    """Resolve one ``int_conditions`` entry.

    Parameters
    ----------
    spec : Sequence or Mapping or Callable
        Values, labeled values, or a function of the observed values.
    values : pandas.Series
        Observed values.

    Returns
    -------
    tuple[list, list[str] or None]
        Values, numeric ones sorted decreasingly as brms does, and their
        labels when given as a mapping.
    """
    if callable(spec):
        resolved: Sequence[ConditionValue] | Mapping[str, ConditionValue] = spec(values)
    else:
        resolved = spec
    if isinstance(resolved, Mapping):
        items = list(resolved.items())
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for _, v in items
        ):
            items.sort(key=lambda kv: -float(cast(float, kv[1])))
        return [v for _, v in items], [k for k, _ in items]
    listed = list(resolved)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in listed):
        listed = sorted(listed, key=lambda v: -float(cast(float, v)))
    return listed, None


def _sort_key(value: ConditionValue, levels: list[ConditionValue] | None) -> tuple:
    """Return R's ``order`` key of a grid value.

    Parameters
    ----------
    value : str or int or float or bool or None
        Value.
    levels : list or None
        Factor levels giving the order of a factor column.

    Returns
    -------
    tuple
        Sortable key.
    """
    if levels is not None and value in levels:
        return (0, levels.index(value), "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def prepare_cond_data(
    context: _Context,
    effects: tuple[str, ...],
    conditions: pd.DataFrame,
    labels: list[str],
    *,
    int_conditions: Mapping[str, IntCondition],
    surface: bool,
    resolution: int,
    reorder: bool,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], list[str] | None]:
    """Build one effect's prediction grid, as brms ``prepare_cond_data``.

    Parameters
    ----------
    context : _Context
        Model frame and variable classes.
    effects : tuple[str, ...]
        One or two variables.
    conditions : pandas.DataFrame
        Completed conditions.
    labels : list[str]
        Condition labels.
    int_conditions : Mapping
        Values of chosen effect variables.
    surface : bool
        Grid a numeric second variable over its range.
    resolution : int
        Numeric grid size.
    reorder : bool
        Put numeric effects first (default effects only).

    Returns
    -------
    tuple[pandas.DataFrame, tuple[str, ...], tuple[str, ...], list[str] or None]
        Grid rows replicated per condition with ``cond__``, the effects and
        their types (``"numeric"``/``"factor"``) in plotting order, and the
        labels of the second effect's ``int_conditions``.
    """
    data = context.data
    types = tuple(
        "factor"
        if _like_factor(_col(data, e)) or e in context.group_vars
        else "numeric"
        for e in effects
    )
    if reorder and len(effects) == 2 and types == ("factor", "numeric"):
        effects, types = effects[::-1], types[::-1]
    values: list[list[ConditionValue] | list[float]] = []
    levels: list[list[ConditionValue] | None] = []
    second_labels: list[str] | None = None
    for position, (effect, kind) in enumerate(zip(effects, types, strict=True)):
        column = _col(data, effect)
        factor_levels = _levels(column) if kind == "factor" else None
        if effect in int_conditions:
            chosen, names = _int_condition(int_conditions[effect], column)
            if position == 1:
                second_labels = names
        elif kind == "factor":
            chosen = list(cast(list[ConditionValue], factor_levels))
        elif position == 0 or surface:
            chosen = list(_numeric_grid(column, effect in context.int_vars, resolution))
        elif effect in context.int_vars:
            median = _number(column.median())
            mad = 1.4826 * _number((column - median).abs().median())
            chosen = [float(round(median + k * mad)) for k in (-1, 0, 1)]
        else:
            mean, sd = _number(column.mean()), _number(column.std(ddof=1))
            chosen = [mean - sd, mean, mean + sd]
        values.append(chosen)
        levels.append(factor_levels)
    if len(effects) == 2:
        rows = [(a, b) for b in values[1] for a in values[0]]
    else:
        rows = [(a,) for a in values[0]]
    unique_rows = list(dict.fromkeys(rows))
    unique_rows.sort(
        key=lambda row: tuple(_sort_key(v, levels[i]) for i, v in enumerate(row))
    )
    grid = pd.DataFrame(unique_rows, columns=list(effects))
    blocks = []
    cond_vars = [c for c in conditions.columns if c not in effects]
    for j, label in enumerate(labels):
        block = grid.copy()
        for v in cond_vars:
            block[v] = [conditions[v].iloc[j]] * len(block)
        block["cond__"] = label
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    frame["cond__"] = pd.Categorical(frame["cond__"], categories=list(labels))
    return frame, effects, types, second_labels


def exclude_too_far(
    g1: np.ndarray, g2: np.ndarray, d1: np.ndarray, d2: np.ndarray, dist: float
) -> np.ndarray:
    """Flag grid points far from the data, as mgcv ``exclude.too.far``.

    Parameters
    ----------
    g1, g2 : numpy.ndarray
        Grid coordinates.
    d1, d2 : numpy.ndarray
        Data coordinates.
    dist : float
        Distance threshold on the grid's unit square.

    Returns
    -------
    numpy.ndarray
        True where the nearest data point is farther than ``dist`` after
        both axes are scaled by the grid's range. Costs O(n m) for n grid
        and m data points.
    """

    def scale(grid: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = grid.min()
        span = (grid - low).max()
        return (grid - low) / span, (data - low) / span

    a1, b1 = scale(np.asarray(g1, float), np.asarray(d1, float))
    a2, b2 = scale(np.asarray(g2, float), np.asarray(d2, float))
    nearest = np.full(a1.shape[0], np.inf)
    for start in range(0, a1.shape[0], 4096):
        stop = start + 4096
        squared = (a1[start:stop, None] - b1[None, :]) ** 2 + (
            a2[start:stop, None] - b2[None, :]
        ) ** 2
        nearest[start:stop] = np.sqrt(squared.min(axis=1))
    return nearest > dist


# --------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------


def _effect_column(frame: pd.DataFrame, effect: str) -> pd.Series:
    """Return the values of one effect, as ``eval2(effect, data)``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Grid or points.
    effect : str
        Variable name.

    Returns
    -------
    pandas.Series
        The column.
    """
    return _col(frame, effect).reset_index(drop=True)


def _second_effect_labels(
    values: pd.Series, labels: list[str] | None
) -> pd.Categorical:
    """Return brms's ``effect2__`` factor of a numeric second effect.

    Parameters
    ----------
    values : pandas.Series
        Second-effect values.
    labels : list[str] or None
        ``int_conditions`` names, used when they match the levels.

    Returns
    -------
    pandas.Categorical
        Values rounded to two decimals, levels in decreasing order.
    """
    rounded = [_r_round(v, 2) for v in values.tolist()]
    order = sorted(set(rounded), reverse=True)
    texts = [_r_text(v) for v in order]
    mapping = dict(zip(order, texts, strict=True))
    if labels is not None and len(labels) == len(order):
        mapping = dict(zip(order, labels, strict=True))
        texts = list(labels)
    return pd.Categorical([mapping[v] for v in rounded], categories=texts)


def _summary(
    draws: torch.Tensor, probs: tuple[float, float], robust: bool
) -> np.ndarray:
    """Summarize draws per row, as brms ``posterior_summary``.

    Parameters
    ----------
    draws : torch.Tensor
        Draws ``(chain, draw, row)``.
    probs : tuple[float, float]
        Lower and upper quantile probabilities.
    robust : bool
        Median and MAD instead of mean and SD.

    Returns
    -------
    numpy.ndarray
        ``(row, 4)``: estimate, error, lower, upper.
    """
    pooled = draws.reshape(-1, *draws.shape[2:]).double()
    return _statistics(pooled, probs, robust).T.numpy()


def _points(
    context: _Context,
    response: pd.Series,
    effects: tuple[str, ...],
    conditions: pd.DataFrame,
    labels: list[str],
    *,
    select_points: float,
    binary: bool,
) -> pd.DataFrame:
    """Assign observed data to conditions, as brms ``make_point_frame``.

    Parameters
    ----------
    context : _Context
        Model frame and variable classes.
    response : pandas.Series
        Observed response.
    effects : tuple[str, ...]
        Plotted variables.
    conditions : pandas.DataFrame
        Completed conditions.
    labels : list[str]
        Condition labels.
    select_points : float
        Maximum unit-scaled distance of a numeric variable to its condition;
        zero ignores numeric variables.
    binary : bool
        Whether the family is binary (response codes start at zero).

    Returns
    -------
    pandas.DataFrame
        Effect columns, ``resp__``, ``cond__`` and ``effect<i>__``.

    Raises
    ------
    ValueError
        If ``select_points`` is negative.
    """
    if select_points < 0:
        raise ValueError("select_points must be nonnegative")
    data = context.data
    effects = tuple(e for e in effects if e in data.columns)
    points = cast(pd.DataFrame, data[list(effects)]).copy()
    points["resp__"] = response.to_numpy()
    groups = set(context.group_vars)
    candidates = [c for c in context.variables if c not in effects]
    candidates += [
        g for g in context.group_vars if g not in candidates and g not in effects
    ]
    required = [c for c in candidates if c in conditions.columns]
    if required:
        blocks = []
        for j, label in enumerate(labels):
            cond = {v: conditions[v].iloc[j] for v in required}
            keep_vars = [
                v
                for v in required
                if cond[v] is not None
                and not (isinstance(cond[v], float) and math.isnan(cond[v]))
                and cond[v] != "zero__"
            ]
            mask = np.ones(len(data), dtype=bool)
            for v in keep_vars:
                column = _col(data, v)
                numeric = not _like_factor(column) and v not in groups
                if numeric:
                    if select_points == 0:
                        continue
                    low, high = _number(column.min()), _number(column.max())
                    unit = (column.to_numpy(float) - low) / (high - low)
                    target = (float(cast(float, cond[v])) - low) / (high - low)
                    mask &= np.abs(unit - target) <= select_points
                else:
                    mask &= column.astype(object).to_numpy() == cond[v]
            block = points.loc[mask].copy()
            block["cond__"] = label
            blocks.append(block)
        points = cast(pd.DataFrame, pd.concat(blocks, ignore_index=True))
        points["cond__"] = pd.Categorical(points["cond__"], categories=list(labels))
    for i, effect in enumerate(effects, start=1):
        points[f"effect{i}__"] = _col(points, effect).to_numpy()
    if not pd.api.types.is_numeric_dtype(
        points["resp__"]
    ) or pd.api.types.is_bool_dtype(points["resp__"]):
        codes = pd.Categorical(points["resp__"]).codes.astype(float) + 1.0
        points["resp__"] = codes - 1.0 if binary else codes
    return cast(pd.DataFrame, points.reset_index(drop=True))


def _spaghetti(
    frame: pd.DataFrame, draws: torch.Tensor, effects: tuple[str, ...]
) -> pd.DataFrame:
    """Return one row per draw and grid row, as brms's spaghetti frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Grid rows.
    draws : torch.Tensor
        Draws ``(chain, draw, row)``.
    effects : tuple[str, ...]
        Plotted variables.

    Returns
    -------
    pandas.DataFrame
        Draw-major rows with ``estimate__`` and ``sample__``; samples are
        1-based draw numbers, suffixed by the second effect's value.
    """
    pooled = draws.reshape(-1, draws.shape[-1]).double().numpy()
    count, rows = pooled.shape
    replicated = pd.concat([frame] * count, ignore_index=True)
    sample = np.repeat(np.arange(1, count + 1), rows).astype(str)
    if len(effects) == 2:
        second = np.tile(
            np.asarray([_r_text(v) for v in frame[effects[1]].tolist()]), count
        )
        sample = np.char.add(np.char.add(sample, "_"), second)
    replicated["estimate__"] = pooled.reshape(-1)
    replicated["sample__"] = pd.Categorical(
        sample, categories=list(dict.fromkeys(sample))
    )
    return replicated


def _requested_effects(
    effects: str | Sequence[str] | None,
    available: list[tuple[str, ...]],
    combined: list[str],
) -> tuple[list[tuple[str, ...]], bool]:
    """Resolve the ``effects`` argument against the model's effects.

    Parameters
    ----------
    effects : str or Sequence[str] or None
        Requested effects such as ``"x"`` or ``"x:z"``; None for defaults.
    available : list[tuple[str, ...]]
        brms's default effects.
    combined : list[str]
        Every effect variable and grouping variable, any pair of which is valid.

    Returns
    -------
    tuple[list[tuple[str, ...]], bool]
        Effects and whether they are brms's defaults.

    Raises
    ------
    ValueError
        If an effect has more than two variables or none is valid.
    """
    if effects is None:
        return available, True
    requested = [effects] if isinstance(effects, str) else list(effects)
    parsed = [tuple(p.strip() for p in e.split(":")) for e in requested]
    if any(len(e) > 2 for e in parsed):
        raise ValueError(
            "to display interactions of order higher than 2, use the conditions argument"
        )
    valid = [(v,) for v in combined] + [
        tuple(sorted(pair)) for pair in combinations(combined, 2)
    ]
    chosen = []
    for effect in parsed:
        if (effect if len(effect) == 1 else tuple(sorted(effect))) in valid:
            if effect not in chosen:
                chosen.append(effect)
    invalid = [":".join(e) for e in parsed if e not in chosen]
    if not chosen:
        raise ValueError(
            "all specified effects are invalid for this model; valid effects are "
            f"(combinations of): {', '.join(combined)}"
        )
    if invalid:
        raise ValueError(
            f"effects {invalid} are invalid for this model; valid effects are "
            f"(combinations of): {', '.join(combined)}"
        )
    return chosen, False


def _draw_ids(
    fit: Fit, ndraws: int | None, draw_ids: Sequence[int] | None, seed: int
) -> list[int] | None:
    """Resolve brms's ``ndraws`` and ``draw_ids``.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    ndraws : int or None
        Random draws to use.
    draw_ids : Sequence[int] or None
        Explicit 0-based draws.
    seed : int
        Seed of the random selection.

    Returns
    -------
    list[int] or None
        Selected draws, or None for all.

    Raises
    ------
    ValueError
        If both are given or ``ndraws`` exceeds the available draws.
    """
    if ndraws is not None and draw_ids is not None:
        raise ValueError("give ndraws or draw_ids, not both")
    if draw_ids is not None:
        return list(draw_ids)
    if ndraws is None:
        return None
    available = fit.ndraws()
    if (
        isinstance(ndraws, bool)
        or not isinstance(ndraws, int)
        or not 0 < ndraws <= available
    ):
        raise ValueError(f"ndraws must be an integer in [1, {available}]")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(available, generator=generator)[:ndraws].tolist()


def _predict(
    fit: Fit,
    response: str,
    frame: pd.DataFrame,
    *,
    method: PredictionMethod,
    dpar: str | None,
    nlpar: str | None,
    re_formula: str | None,
    draw_ids: list[int] | None,
    seed: int,
    mc_samples: int | None = None,
) -> torch.Tensor:
    """Predict grid rows through the fit's prediction methods.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    response : str
        Joint response, or empty for a single-response fit.
    frame : pandas.DataFrame
        Grid rows.
    method : {"posterior_epred", "posterior_linpred", "posterior_predict"}
        brms prediction method.
    dpar, nlpar : str or None
        Predicted distributional or nonlinear parameter.
    re_formula : str or None
        Varying effects included (``"~0"`` excludes all, brms's ``NA``).
    draw_ids : list[int] or None
        Selected draws.
    seed : int
        Seed of predictions.
    mc_samples : int or None
        Monte Carlo budget of a conditional mean without a closed form, as
        `BayesianFit.posterior_epred` takes it.

    Returns
    -------
    torch.Tensor
        Draws ``(chain, draw, row[, category])``.

    Raises
    ------
    ValueError
        If a parameter is requested with ``posterior_predict``, or
        ``mc_samples`` is given for another method than ``posterior_epred``
        or for a joint fit.
    """
    if mc_samples is not None and (method != "posterior_epred" or _joint(fit)):
        raise ValueError(
            "mc_samples applies only to epred effects of a single-response fit"
        )
    parameter = dpar or nlpar or "mu"
    if parameter != "mu" and method == "posterior_predict":
        raise ValueError("dpar and nlpar require posterior_epred or posterior_linpred")
    if parameter != "mu":
        # brms posterior_epred(dpar=) is the parameter on its own scale.
        transform = method == "posterior_epred" and nlpar is None
        method = "posterior_linpred"
    else:
        transform = False
    columns = frame.drop(columns=[c for c in frame.columns if c.endswith("__")])
    if _joint(fit):
        options: JointPredictionOptions = {
            "new_data": columns,
            "re_formula": re_formula,
            "allow_new_levels": True,
            "seed": seed,
        }
        return cast("MultivariateFit", fit)._pp_prediction(
            method,
            response,
            options,
            draw_ids=draw_ids,
            parameter=parameter,
            transform=transform,
        )[response]
    single: SinglePredictionOptions = {
        "new_data": columns,
        "re_formula": re_formula,
        "allow_new_levels": True,
        "incl_autocor": False,
        "seed": seed,
    }
    view = cast("BayesianFit", fit)
    if (
        view.family.trials_parameter is not None
        and dict(view.formula.modifiers.sources).get("trials") is None
        and view.binomial_trials is None
    ):
        # A count-vector response whose totals were read from its rows has
        # no total on a grid. brms sets the trials of effects to one, which
        # gives the category probabilities.
        single["trials"] = 1
    if mc_samples is not None:
        # The Monte Carlo budget is a keyword of posterior_epred alone.
        single = cast(SinglePredictionOptions, {**single, "mc_samples": mc_samples})
    return cast("BayesianFit", fit)._pp_prediction(
        method, single, draw_ids=draw_ids, parameter=parameter, transform=transform
    )


def brms_conditional_effects(
    fit: Fit,
    effects: str | Sequence[str] | None = None,
    *,
    conditions: IntoDataFrame | Mapping[str, Sequence[ConditionValue]] | None = None,
    int_conditions: Mapping[str, IntCondition] | None = None,
    re_formula: str | None = "~0",
    prob: float = 0.95,
    robust: bool = True,
    method: PredictionMethod = "posterior_epred",
    spaghetti: bool = False,
    surface: bool = False,
    categorical: bool = False,
    ordinal: bool = False,
    resolution: int = 100,
    select_points: float = 0.0,
    too_far: float = 0.0,
    resp: str | Sequence[str] | None = None,
    dpar: str | None = None,
    nlpar: str | None = None,
    ndraws: int | None = None,
    draw_ids: Sequence[int] | None = None,
    seed: int = 0,
    mc_samples: int | None = None,
) -> BrmsConditionalEffects:
    """Compute brms's automatic conditional effects of a fit.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model with retained training data.
    effects : str or Sequence[str] or None
        Effects such as ``"x"`` or ``"x:z"``; None selects brms's defaults.
    conditions : IntoDataFrame or Mapping or None
        Rows of values of other variables, each one condition, optionally
        labeled by ``cond__`` (see `make_conditions`).
    int_conditions : Mapping or None
        Values (a sequence, a labeled mapping, or a function of the data) of
        effect variables, replacing brms's default grid values.
    re_formula : str or None
        Varying effects included: ``"~0"`` (brms's ``NA``) none, None all.
    prob : float
        Interval probability.
    robust : bool
        Median and MAD (brms's default) instead of mean and SD.
    method : {"posterior_epred", "posterior_linpred", "posterior_predict"}
        Prediction method.
    spaghetti : bool
        Also return per-draw lines of numeric first effects.
    surface : bool
        Grid a numeric second effect over its range.
    categorical : bool
        Per-category probabilities of categorical and ordinal models.
    ordinal : bool
        brms's deprecated alias of ``categorical``.
    resolution : int
        Numeric grid size.
    select_points : float
        Unit-scaled distance for assigning data points to numeric conditions.
    too_far : float
        Drop surface grid points farther than this from the data.
    resp : str or Sequence[str] or None
        Responses of a joint fit.
    dpar, nlpar : str or None
        Distributional or nonlinear parameter to predict instead of the mean.
    ndraws : int or None
        Random subset of draws.
    draw_ids : Sequence[int] or None
        Explicit 0-based draws.
    seed : int
        Seed of draw selection and predictions.
    mc_samples : int or None
        Monte Carlo budget of a conditional mean without a closed form
        (``posterior_epred`` of a single-response fit only).

    Returns
    -------
    BrmsConditionalEffects
        One `EffectFrame` per effect (per response for joint fits).

    Raises
    ------
    ValueError
        If arguments are invalid, effects are invalid for the model, the fit
        retains no data, or a categorical model is summarized without
        ``categorical=True``.

    Notes
    -----
    Each effect costs one prediction over ``G C`` rows for ``G`` grid points
    and ``C`` conditions, O(S G C) for S draws beyond the model's own
    per-row prediction cost.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    if method not in {"posterior_epred", "posterior_linpred", "posterior_predict"}:
        raise ValueError(
            "method must be posterior_epred, posterior_linpred, or posterior_predict"
        )
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution < 1
    ):
        raise ValueError("resolution must be a positive integer")
    if categorical and ordinal:
        raise ValueError("use categorical instead of ordinal")
    per_category = categorical or ordinal
    if spaghetti and surface:
        raise ValueError("spaghetti and surface cannot be combined")
    probs = ((1.0 - prob) / 2.0, 1.0 - (1.0 - prob) / 2.0)
    # Effects and conditions come from every response; resp selects predictions.
    every = _views(fit, None)
    views = _views(fit, resp)
    data = model_frame(fit)
    variables = model_variables(every, data)
    context = _Context(
        data=data,
        variables=tuple(variables),
        group_vars=tuple(sorted(_group_variables(every, data))),
        int_vars=tuple(sorted(_integer_variables(every))),
    )
    available: list[tuple[str, ...]] = []
    for view in every.values():
        for effect in _parameter_effects(view):
            if effect not in available:
                available.append(effect)
    combined = list(dict.fromkeys(v for e in available for v in e))
    combined += [g for g in context.group_vars if g not in combined]
    chosen, defaults = _requested_effects(effects, available, combined)
    if per_category:
        chosen = [e for e in chosen if len(e) == 1]
    if not chosen:
        raise ValueError("no valid effects detected")
    filled, labels = prepare_conditions(context, every, conditions)
    resolved_int = dict(int_conditions or {})
    selected = _draw_ids(fit, ndraws, draw_ids, seed)
    out: list[EffectFrame] = []
    # brms loops effects, then responses (conditional_effects.mvbrmsterms).
    for effect in chosen:
        for response, view in views.items():
            family = view.family
            is_categorical = family.response is not None
            is_ordinal = family.response_ordinal() is not None
            frame, ordered, types, second_labels = prepare_cond_data(
                context,
                effect,
                filled,
                labels,
                int_conditions=resolved_int,
                surface=surface,
                resolution=resolution,
                reorder=defaults,
            )
            is_surface = (
                surface and len(ordered) == 2 and types == ("numeric", "numeric")
            )
            if is_surface and too_far > 0:
                far = exclude_too_far(
                    _col(frame, ordered[0]).to_numpy(float),
                    _col(frame, ordered[1]).to_numpy(float),
                    _col(data, ordered[0]).to_numpy(float),
                    _col(data, ordered[1]).to_numpy(float),
                    too_far,
                )
                frame = frame.loc[~far].reset_index(drop=True)
            draws = _predict(
                fit,
                response,
                frame,
                method=method,
                dpar=dpar,
                nlpar=nlpar,
                re_formula=re_formula,
                draw_ids=selected,
                seed=seed,
                mc_samples=mc_samples,
            ).double()
            names = list(ordered)
            categories: tuple[int | str, ...] = ()
            if per_category:
                if method != "posterior_epred":
                    raise ValueError("categorical requires method='posterior_epred'")
                if not (is_categorical or is_ordinal):
                    raise ValueError(
                        "categorical applies only to categorical or ordinal models"
                    )
                categories = view.category_labels
                frame = pd.concat([frame] * len(categories), ignore_index=True)
                frame["cats__"] = pd.Categorical(
                    np.repeat(
                        [str(c) for c in categories], len(frame) // len(categories)
                    ),
                    categories=[str(c) for c in categories],
                )
                draws = draws.movedim(-1, 2).reshape(*draws.shape[:2], -1)
                names = [names[0], "cats__"]
                types = (types[0], "factor")
            elif dpar is None and nlpar is None:
                if is_categorical:
                    raise ValueError(
                        "set categorical=True for the effects of a categorical model"
                    )
                if is_ordinal and method == "posterior_epred":
                    # brms: equidistant category scores 1..K.
                    scores = torch.arange(1, draws.shape[-1] + 1, dtype=draws.dtype)
                    draws = (draws * scores).sum(-1)
            if draws.ndim != 3:
                raise ValueError(
                    "conditional effects require one prediction per grid row; "
                    "select a scalar parameter"
                )
            for i, name in enumerate(names, start=1):
                frame[f"effect{i}__"] = _effect_column(frame, name)
            if len(names) == 2 and types[1] == "numeric" and not surface:
                frame["effect2__"] = _second_effect_labels(
                    _col(frame, names[1]), second_labels
                )
            spag = (
                _spaghetti(frame, draws, tuple(names))
                if spaghetti and types[0] == "numeric"
                else None
            )
            stats = _summary(draws, probs, robust)
            frame["estimate__"] = stats[:, 0]
            frame["se__"] = stats[:, 1]
            frame["lower__"] = stats[:, 2]
            frame["upper__"] = stats[:, 3]
            observed = view.formula.response_name if not response else response
            response_values = (
                _col(data, observed)
                if observed in data.columns
                else pd.Series(
                    np.asarray(view.formula.response_values).reshape(len(data), -1)[
                        :, 0
                    ]
                )
            )
            points = _points(
                context,
                response_values,
                tuple(names),
                filled,
                labels,
                select_points=select_points,
                binary=family.native.support == "boolean",
            )
            prefix = f"{response}.{response}_" if response else ""
            out.append(
                EffectFrame(
                    name=prefix + ":".join(names),
                    effects=tuple(names),
                    response=dpar
                    or nlpar
                    or view.formula.formula.split("~", 1)[0].strip(),
                    frame=frame,
                    draws=draws,
                    surface=is_surface,
                    categorical=per_category,
                    spaghetti=spag,
                    points=points,
                )
            )
    return BrmsConditionalEffects(effects=tuple(out))


def brms_conditional_smooths(
    fit: Fit,
    smooths: str | Sequence[str] | None = None,
    *,
    resp: str | Sequence[str] | None = None,
    int_conditions: Mapping[str, IntCondition] | None = None,
    prob: float = 0.95,
    spaghetti: bool = False,
    surface: bool = True,
    resolution: int = 100,
    too_far: float = 0.0,
    ndraws: int | None = None,
    draw_ids: Sequence[int] | None = None,
    seed: int = 0,
) -> BrmsConditionalEffects:
    """Compute every smooth's contribution on a grid, as brms ``conditional_smooths``.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model with ``s()`` or ``t2()`` terms and retained data.
    smooths : str or Sequence[str] or None
        Smooth term labels (white space ignored); None for all.
    resp : str or Sequence[str] or None
        Responses of a joint fit whose smooths are evaluated; None for all.
    int_conditions : Mapping or None
        Values of covariates or ``by`` variables.
    prob : float
        Interval probability.
    spaghetti : bool
        Also return per-draw curves of one-dimensional smooths.
    surface : bool
        Grid a numeric second covariate over its range (brms's default).
    resolution : int
        Grid size per numeric covariate.
    too_far : float
        Drop surface grid points farther than this from the data.
    ndraws : int or None
        Random subset of draws.
    draw_ids : Sequence[int] or None
        Explicit 0-based draws.
    seed : int
        Seed of draw selection.

    Returns
    -------
    BrmsConditionalEffects
        One `EffectFrame` named ``<parameter>: <term>`` per smooth
        (``<parameter>_<resp>: <term>`` in a joint fit, as brms), with
        robust summaries (median and MAD) as brms computes them.

    Raises
    ------
    ValueError
        If the fit has no matching smooth term or the arguments are invalid.

    Notes
    -----
    Each smooth costs one evaluation of its contribution over the grid,
    O(S G B) for S draws, G grid points and basis width B.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or resolution < 1
    ):
        raise ValueError("resolution must be a positive integer")
    probs = ((1.0 - prob) / 2.0, 1.0 - (1.0 - prob) / 2.0)
    wanted = (
        None
        if smooths is None
        else {
            re.sub(r"\s", "", s)
            for s in ([smooths] if isinstance(smooths, str) else smooths)
        }
    )
    data = model_frame(fit)
    views = _views(fit, resp)
    context = _Context(
        data=data,
        variables=tuple(model_variables(views, data)),
        group_vars=tuple(sorted(_group_variables(views, data))),
        int_vars=tuple(sorted(_integer_variables(views))),
    )
    filled, _ = prepare_conditions(context, views, None)
    resolved_int = dict(int_conditions or {})
    selected = _draw_ids(fit, ndraws, draw_ids, seed)
    replayed = fit if selected is None else fit._pp_view(selected, seed)
    # A response of a joint fit reads the joint draws under its namespace.
    joint_draws = (
        cast("MultivariateFit", replayed).draws(seed=seed) if _joint(fit) else None
    )
    out: list[EffectFrame] = []
    for response, member in views.items():
        plan = member.prediction_plan
        if plan is None:
            continue
        view = member if _joint(fit) else cast("BayesianFit", replayed)
        site_prefix = (
            cast("MultivariateFit", fit).emission.responses[response].prefix
            if _joint(fit)
            else ""
        )
        prefixes = {name: prefix for name, prefix, _ in view._predictor_items()}
        for parameter, design in plan.designs.items():
            if design.common is None:
                continue
            kinds = _effect_kinds(view, prefixes[parameter])
            for label, term in design.common.terms.items():
                if kinds.get(label) != "sm":
                    continue
                compact = re.sub(r"\s", "", label)
                if wanted is not None and compact not in wanted:
                    continue
                sources = design_variables(term)
                byvars = list(sources.by)
                covars = [v for v in sources.names if v not in byvars]
                if len(covars) > 2:
                    byvars = covars[2:] + byvars
                    covars = covars[:2]
                elif len(covars) == 1 and byvars:
                    covars = covars + byvars[:1]
                    byvars = byvars[1:]
                grid_values: list[list[ConditionValue] | list[float]] = []
                numeric = []
                for position, cv in enumerate(covars):
                    column = _col(data, cv)
                    is_numeric = not _like_factor(column)
                    numeric.append(is_numeric)
                    if cv in resolved_int:
                        grid_values.append(_int_condition(resolved_int[cv], column)[0])
                    elif is_numeric:
                        if not surface and position == 1:
                            mean, sd = (
                                _number(column.mean()),
                                _number(column.std(ddof=1)),
                            )
                            grid_values.append([mean - sd, mean, mean + sd])
                        else:
                            grid_values.append(
                                list(_numeric_grid(column, False, resolution))
                            )
                    else:
                        grid_values.append(_levels(column))
                for cv in byvars:
                    column = _col(data, cv)
                    if cv in resolved_int:
                        grid_values.append(_int_condition(resolved_int[cv], column)[0])
                    elif not _like_factor(column):
                        mean, sd = _number(column.mean()), _number(column.std(ddof=1))
                        grid_values.append([mean - sd, mean, mean + sd])
                    else:
                        grid_values.append(_levels(column))
                variables = covars + byvars
                index = pd.MultiIndex.from_product(
                    grid_values[::-1], names=variables[::-1]
                )
                grid = cast(pd.DataFrame, index.to_frame(index=False)[variables])
                show_surface = surface and len(covars) == 2 and all(numeric)
                if show_surface and too_far > 0:
                    far = exclude_too_far(
                        _col(grid, covars[0]).to_numpy(float),
                        _col(grid, covars[1]).to_numpy(float),
                        _col(data, covars[0]).to_numpy(float),
                        _col(data, covars[1]).to_numpy(float),
                        too_far,
                    )
                    grid = grid.loc[~far].reset_index(drop=True)
                newdata = grid.copy()
                for v in filled.columns:
                    if v not in newdata.columns:
                        newdata[v] = [filled[v].iloc[0]] * len(newdata)
                eta = smooth_contribution(
                    view,
                    label,
                    newdata,
                    parameter=parameter,
                    seed=seed,
                    draws=joint_draws,
                    site_prefix=site_prefix,
                )
                output = next(
                    item.output
                    for item in plan.predictors
                    if item.parameter == parameter
                )
                # A vector-valued parameter has one effect per coordinate.
                parameters = output.parameters if output is not None else (parameter,)
                effects = eta.unbind(-1) if output is not None else (eta,)
                for coordinate, values in zip(parameters, effects, strict=True):
                    name = f"{coordinate}_{response}" if response else coordinate
                    frame = cast(pd.DataFrame, grid[variables]).copy()
                    plotted = covars[:2]
                    for i, covariate in enumerate(plotted, start=1):
                        frame[f"effect{i}__"] = _col(frame, covariate).to_numpy()
                    if len(numeric) == 2 and numeric[1] and not surface:
                        frame["effect2__"] = _second_effect_labels(
                            _col(frame, plotted[1]), None
                        )
                    frame["cond__"] = (
                        pd.Categorical(rows2labels(cast(pd.DataFrame, frame[byvars])))
                        if byvars
                        else pd.Categorical(["1"] * len(frame))
                    )
                    spag = (
                        _spaghetti(frame, values, tuple(plotted))
                        if spaghetti and not show_surface and numeric[0]
                        else None
                    )
                    stats = _summary(values.double(), probs, True)
                    frame["estimate__"] = stats[:, 0]
                    frame["se__"] = stats[:, 1]
                    frame["lower__"] = stats[:, 2]
                    frame["upper__"] = stats[:, 3]
                    points = cast(pd.DataFrame, data[variables]).copy()
                    for i, covariate in enumerate(covars, start=1):
                        points[f"effect{i}__"] = points[covariate].to_numpy()
                    out.append(
                        EffectFrame(
                            name=f"{name}: {compact}",
                            effects=tuple(plotted),
                            response=f"{name}: {compact}",
                            frame=frame,
                            draws=values.double(),
                            surface=show_surface,
                            spaghetti=spag,
                            points=points,
                        )
                    )
    if not out:
        raise ValueError("no valid smooth terms found in the model")
    return BrmsConditionalEffects(effects=tuple(out), smooths_only=True)


class ConditionalEffectsOptions(ExpectationOptions, total=False):
    """Keywords of the fits' ``conditional_effects`` methods.

    Prediction keys follow `BayesianFit.posterior_epred` and apply to explicit
    prediction grids. The remaining keys are brms ``conditional_effects``
    arguments of automatic effects (see `brms_conditional_effects`).
    """

    conditions: IntoDataFrame | Mapping[str, Sequence[ConditionValue]] | None
    int_conditions: Mapping[str, IntCondition] | None
    prob: float
    robust: bool
    spaghetti: bool
    surface: bool
    categorical: bool
    ordinal: bool
    resolution: int
    select_points: float
    too_far: float
    dpar: str | None
    nlpar: str | None
    ndraws: int | None
    draw_ids: Sequence[int] | None


class JointConditionalEffectsOptions(TypedDict, total=False):
    """Keywords of `MultivariateFit.conditional_effects`.

    ``re_formula``, ``allow_new_levels``, ``sample_new_levels``,
    ``num_samples`` and ``seed`` follow `MultivariateFit.posterior_predict`
    and apply to explicit grids; the remaining keys are brms
    ``conditional_effects`` arguments of automatic effects.
    """

    re_formula: str | None
    allow_new_levels: bool
    sample_new_levels: NewLevelPolicy
    num_samples: int | None
    seed: int
    conditions: IntoDataFrame | Mapping[str, Sequence[ConditionValue]] | None
    int_conditions: Mapping[str, IntCondition] | None
    prob: float
    robust: bool
    spaghetti: bool
    surface: bool
    categorical: bool
    ordinal: bool
    resolution: int
    select_points: float
    too_far: float
    dpar: str | None
    nlpar: str | None
    ndraws: int | None
    draw_ids: Sequence[int] | None


def joint_grid_options(
    options: JointConditionalEffectsOptions,
) -> JointPredictionOptions:
    """Return the prediction keywords of an explicit joint-grid request.

    Parameters
    ----------
    options : JointConditionalEffectsOptions
        Method keywords.

    Returns
    -------
    JointPredictionOptions
        The prediction keywords.

    Raises
    ------
    ValueError
        If a brms automatic-effects keyword is given with an explicit grid.
    """
    extra = sorted(_AUTOMATIC_KEYS & set(options))
    if extra:
        raise ValueError(
            f"{extra} apply to automatic conditional effects, not to an explicit grid"
        )
    return cast(JointPredictionOptions, options)


#: brms-only keys, refused for explicit prediction grids.
_AUTOMATIC_KEYS = frozenset(ConditionalEffectsOptions.__annotations__) - frozenset(
    ExpectationOptions.__annotations__
)

#: Short method names of explicit grids and brms's prediction function names.
_METHODS: dict[str, PredictionMethod] = {
    "epred": "posterior_epred",
    "linpred": "posterior_linpred",
    "predict": "posterior_predict",
    "posterior_epred": "posterior_epred",
    "posterior_linpred": "posterior_linpred",
    "posterior_predict": "posterior_predict",
}


def grid_options(options: ConditionalEffectsOptions) -> ExpectationOptions:
    """Return the prediction keywords of an explicit-grid request.

    Parameters
    ----------
    options : ConditionalEffectsOptions
        Method keywords.

    Returns
    -------
    ExpectationOptions
        The same keywords.

    Raises
    ------
    ValueError
        If a brms automatic-effects keyword is given with an explicit grid.
    """
    extra = sorted(_AUTOMATIC_KEYS & set(options))
    if extra:
        raise ValueError(
            f"{extra} apply to automatic conditional effects, not to an explicit grid"
        )
    return cast(ExpectationOptions, options)


def automatic_conditional_effects(
    fit: Fit,
    effects: str | Sequence[str] | None,
    *,
    method: str,
    parameter: str,
    transform: bool,
    resp: str | Sequence[str] | None,
    options: ConditionalEffectsOptions | JointConditionalEffectsOptions,
) -> BrmsConditionalEffects:
    """Dispatch a fit's ``conditional_effects`` call without a grid.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    effects : str or Sequence[str] or None
        brms ``effects``.
    method : str
        ``epred``/``linpred``/``predict`` or brms's function names.
    parameter : str
        ``mu`` or a distributional parameter (brms ``dpar``).
    transform : bool
        Must be False; brms applies no inverse link to ``posterior_linpred``.
    resp : str or Sequence[str] or None
        Joint responses.
    options : ConditionalEffectsOptions or JointConditionalEffectsOptions
        brms arguments and ``re_formula`` and ``seed``.

    Returns
    -------
    BrmsConditionalEffects
        The effects.

    Raises
    ------
    ValueError
        If the method is unknown, or a keyword applies only to explicit grids.
    """
    if method not in _METHODS:
        raise ValueError(
            "method must be epred, linpred, predict, posterior_epred, "
            "posterior_linpred, or posterior_predict"
        )
    unsupported = sorted(
        set(options) - _AUTOMATIC_KEYS - {"re_formula", "seed", "mc_samples"}
    )
    if unsupported or transform:
        raise ValueError(
            f"{unsupported or ['transform']} apply only to explicit prediction grids; "
            "automatic effects choose rows and draws themselves (use ndraws)"
        )
    dpar = options.get("dpar")
    if parameter != "mu":
        if dpar is not None and dpar != parameter:
            raise ValueError("give the parameter once, as parameter or dpar")
        dpar = parameter
    return brms_conditional_effects(
        fit,
        effects,
        conditions=options.get("conditions"),
        int_conditions=options.get("int_conditions"),
        re_formula=options.get("re_formula", "~0"),
        prob=options.get("prob", 0.95),
        robust=options.get("robust", True),
        method=_METHODS[method],
        spaghetti=options.get("spaghetti", False),
        surface=options.get("surface", False),
        categorical=options.get("categorical", False),
        ordinal=options.get("ordinal", False),
        resolution=options.get("resolution", 100),
        select_points=options.get("select_points", 0.0),
        too_far=options.get("too_far", 0.0),
        resp=resp,
        dpar=dpar,
        nlpar=options.get("nlpar"),
        ndraws=options.get("ndraws"),
        draw_ids=options.get("draw_ids"),
        seed=options.get("seed", 0),
        mc_samples=options.get("mc_samples"),
    )


__all__ = [
    "BrmsConditionalEffects",
    "ConditionalEffectsOptions",
    "automatic_conditional_effects",
    "grid_options",
    "EffectFrame",
    "EffectMethod",
    "JointConditionalEffectsOptions",
    "joint_grid_options",
    "brms_conditional_effects",
    "brms_conditional_smooths",
    "exclude_too_far",
    "make_conditions",
    "model_frame",
    "prepare_conditions",
    "rows2labels",
]
