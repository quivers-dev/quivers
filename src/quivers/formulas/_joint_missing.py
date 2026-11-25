"""brms ``mi()`` missing-value terms in a joint formula program.

A response ``x | mi() ~ ...`` keeps its ordinary family law. Its recorded
values are observed, and each missing value is a latent coordinate drawn from
the same law at its row, exactly the joint density brms writes as
``normal_lpdf(Yl | mu, sigma)`` over ``Yl`` with ``Yl[Jmi] = Ymi``. With known
measurement error, ``x | mi(sdx) ~ ...``, every true value is latent and each
recorded value is a Normal measurement of it, as in brms. A predictor term
``mi(x)`` (or ``mi(x, idx)``) uses the completed values of that response; it
enters the linear predictor linearly, also inside interactions and as the
slope of a group-level term ``(mi(x) | g)``, whose varying coefficient
multiplies the completed values.

The emitted program is the single-response emission of every response,
rewritten once after joint composition: each ``mi()`` response's observation
is split into recorded and latent coordinates by explicit index inputs, and
each ``mi()`` design column becomes a `SymbolicColumn` bound to the completed
values. No column is filled with an estimate.

Under estimated residual correlations the row's responses are one vector
event. brms scores the completed vectors under the joint law,
``multi_normal_cholesky_lpdf(Yl | Mu, LSigma)``, with the missing values as
parameters without a prior. The rewrite then draws each missing value (with
measurement error, each true value) from the improper flat law and replaces
the vector observation by one log-density factor over the completed vectors.
Prediction programs keep the vector observation, whose law is read at the
replayed completed values.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import replace

import didactic.api as dx
from formulae.matrices import DesignMatrices
import numpy as np
import pandas as pd
import torch

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    ExportDecl,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    ObserveStep,
    OptionName,
    ProgramDecl,
    ProgramStep,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
    SampleStep,
    ScoreStep,
    Statement,
    TypeName,
)
from quivers.dsl.program_templates import draw_argument_expression
from quivers.formulas._ast import (
    _add,
    _call,
    _div,
    _factor,
    _index,
    _let,
    _mul,
    _sub,
    _var,
)
from quivers.formulas._brms_priors import flat_prior_step
from quivers.formulas._design import SymbolicColumn
from quivers.formulas._fit import _PreparedFormula
from quivers.formulas._joint_compile import _JointEmission, _ResponseEmission
from quivers.formulas._missing import (
    LatentResponseSpec,
    bind_latent_input,
    bind_latent_response,
)
from quivers.formulas._multivariate import MultivariateFormula, ResponseModel
from quivers.formulas._predictor import data_terms
from quivers.formulas._row_source import RowSource, finite_inputs
from quivers.formulas._term import bind_inputs
from quivers.formulas._prior_emission import order_prior_dependencies
from quivers.formulas._priors import PriorTarget
from quivers.formulas.compile import FormulaToQVRModule
from quivers.formulas.formula import (
    FixedColumn,
    Formula,
    RandomTerm,
    RowSelection,
    _qvr_name,
)
from quivers.qiec import NAT, REAL, IndexLiteral, ProgramParameter
from quivers.qiec.canonical import tensor_type

__all__: list[str] = []

_MEASUREMENT = re.compile(
    r"\bme\(\s*([A-Za-z_][\w.]*)\s*,\s*([A-Za-z_][\w.]*)\s*"
    r"(?:,\s*(?:gr\s*=\s*)?([A-Za-z_][\w.]*)\s*)?\)"
)
_TERM = re.compile(r"\bmi\(\s*([A-Za-z_][\w.]*)\s*(?:,\s*([A-Za-z_][\w.]*)\s*)?\)")


def _split_top(text: str, separator: str) -> list[str]:
    """Split text at separators outside parentheses.

    Parameters
    ----------
    text : str
        Formula fragment.
    separator : str
        One-character separator.

    Returns
    -------
    list[str]
        Stripped parts.
    """
    parts, depth, current = [], 0, ""
    for character in text:
        depth += (character == "(") - (character == ")")
        if character == separator and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += character
    parts.append(current.strip())
    return parts


class MissingResponse(ValueModel):
    """One response declared with ``mi()`` on its left side.

    Parameters
    ----------
    label : str
        Response label.
    variable : str
        Response data column.
    noise : str or None
        Column of known measurement standard deviations, ``mi(sdy)``.
    index : str or None
        Identity column of an ``index()`` addition term, used by ``mi(x, idx)``.
    """

    label: str
    variable: str
    noise: str | None = None
    index: str | None = None


class MissingInput(ValueModel):
    """One design column of a consumer predictor that uses ``mi()`` values.

    Parameters
    ----------
    consumer : str
        Consumer response label.
    parameter : str
        Predictor of that response (``mu``, a distributional or nonlinear name).
    prefix : str
        Local coefficient prefix of the predictor.
    column : FixedColumn
        Evaluated column with every ``mi()`` variable set to one, retaining
        the column's other factors as its data.
    sources : tuple[str, ...]
        Labels of the ``mi()`` responses whose completed values multiply it.
    indices : tuple[str or None, ...]
        Consumer ``idx`` column for each source, or None for a row join.
    group : str or None
        Grouping factor of a varying slope whose column this is, or None for
        a population-level column.
    slope : str
        Slope label of that varying slope.
    """

    consumer: str
    parameter: str
    prefix: str
    column: FixedColumn = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    sources: tuple[str, ...]
    indices: tuple[str | None, ...]
    group: str | None = None
    slope: str = ""

    @property
    def input_name(self) -> str:
        """Name the consumer's local input this column fills.

        Returns
        -------
        str
            ``<prefix><column>``, or ``<prefix>random_<column>`` for the
            column of a varying slope.
        """
        middle = "" if self.group is None else "random_"
        return f"{self.prefix}{middle}{self.column.qvr_name}"

    @property
    def binding(self) -> str:
        """Name the joint binding holding this column's values.

        Returns
        -------
        str
            ``mi_<consumer>__<local input>``.
        """
        return f"mi_{self.consumer}__{self.input_name}"


class MeasurementTerm(ValueModel):
    """One ``me(x, sdx, gr)`` measurement-error predictor of a response.

    Parameters
    ----------
    key : str
        Joint identity, ``<consumer>__<variable>``.
    consumer : str
        Response whose predictor uses the term.
    variable : str
        Column of noisy recorded values.
    noise : str
        Column of known measurement standard deviations.
    group : str or None
        Grouping factor sharing one latent value per level (``gr``).
    label : str
        brms coefficient label, ``me`` followed by the variable and noise
        columns with non-alphanumeric characters removed.
    latent : str
        brms latent-value label, ``me`` followed by the variable, used by
        ``meanme``, ``sdme``, and ``corme``.
    """

    key: str
    consumer: str
    variable: str
    noise: str
    group: str | None = None
    label: str
    latent: str


class MissingPlan(ValueModel):
    """Every ``mi()`` and ``me()`` declaration of one joint formula.

    Parameters
    ----------
    responses : dict[str, MissingResponse]
        ``mi()`` responses by label.
    inputs : tuple[MissingInput, ...]
        Consumer design columns using completed or latent values.
    measurements : dict[str, MeasurementTerm]
        ``me()`` terms by key.
    mecor : dict[str, bool]
        brms ``set_mecor`` setting of each consumer: correlate the latent
        values of its ``me()`` terms that share a grouping.
    """

    responses: dict[str, MissingResponse] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    inputs: tuple[MissingInput, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    measurements: dict[str, MeasurementTerm] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_VALUE
    )
    mecor: dict[str, bool] = dx.field(default_factory=dict)

    @property
    def empty(self) -> bool:
        """Report whether the formula has no ``mi()`` or ``me()`` term.

        Returns
        -------
        bool
            True without ``mi()`` responses and ``me()`` terms.
        """
        return not self.responses and not self.measurements


def _rewrite_response(
    label: str, response: ResponseModel
) -> tuple[
    ResponseModel,
    MissingResponse | None,
    list[tuple[str, str | None]],
    list[tuple[str, str, str | None]],
]:
    """Remove ``mi()`` and ``me()`` syntax from one response.

    Parameters
    ----------
    label : str
        Response label.
    response : ResponseModel
        Authored response.

    Returns
    -------
    tuple
        The response with ordinary syntax, its ``mi()`` declaration or None,
        the ``mi()`` and ``me()`` predictor variables with their index
        columns, and each ``me()`` term's variable, noise, and grouping.

    Raises
    ------
    ValueError
        If ``mi()`` names more than one column, or ``index()`` appears without
        ``mi()``.
    """
    left, right = response.formula.split("~", 1)
    declaration = None
    if "|" in left:
        variable, terms = left.split("|", 1)
        kept, noise, index, flagged = [], None, None, False
        for term in _split_top(terms, "+"):
            call = re.fullmatch(r"(mi|index)\s*\((.*)\)", term, re.S)
            if call is None:
                kept.append(term)
                continue
            arguments = [a.strip() for a in _split_top(call.group(2), ",") if a.strip()]
            if call.group(1) == "mi":
                if len(arguments) > 1:
                    raise ValueError(
                        f"response {label!r}: mi() takes at most one measurement-error column"
                    )
                flagged, noise = True, (arguments[0] if arguments else None)
            else:
                if len(arguments) != 1:
                    raise ValueError(f"response {label!r}: index() names one column")
                index = arguments[0]
        if index is not None and not flagged:
            raise ValueError(
                f"response {label!r}: index() identifies rows of an mi() response"
            )
        if flagged:
            declaration = MissingResponse(
                label=label, variable=variable.strip(), noise=noise, index=index
            )
            left = variable + (" | " + " + ".join(kept) if kept else " ")
    uses: list[tuple[str, str | None]] = []

    def replace_term(match: re.Match[str]) -> str:
        """Record one ``mi()`` predictor and keep its plain column.

        Parameters
        ----------
        match : re.Match[str]
            One ``mi(x)`` or ``mi(x, idx)`` occurrence.

        Returns
        -------
        str
            The column name.
        """
        uses.append((match.group(1), match.group(2)))
        return match.group(1)

    measured: list[tuple[str, str, str | None]] = []

    def replace_measurement(match: re.Match[str]) -> str:
        """Record one ``me()`` predictor and keep its plain column.

        Parameters
        ----------
        match : re.Match[str]
            One ``me(x, sdx)`` or ``me(x, sdx, gr = g)`` occurrence.

        Returns
        -------
        str
            The column name.
        """
        measured.append((match.group(1), match.group(2), match.group(3)))
        uses.append((match.group(1), None))
        return match.group(1)

    right = _MEASUREMENT.sub(replace_measurement, right)
    formula = left + "~" + _TERM.sub(replace_term, right)
    distributional = (
        {
            key: _TERM.sub(replace_term, value) if isinstance(value, str) else value
            for key, value in response.distributional.items()
        }
        if response.distributional is not None
        else None
    )
    nonlinear = (
        {
            key: _TERM.sub(replace_term, value) if isinstance(value, str) else value
            for key, value in response.nonlinear.items()
        }
        if response.nonlinear is not None
        else None
    )
    return (
        response.with_(
            formula=formula, distributional=distributional, nonlinear=nonlinear
        ),
        declaration,
        uses,
        measured,
    )


def missing_specification(
    specification: MultivariateFormula,
) -> tuple[
    MultivariateFormula,
    dict[str, MissingResponse],
    dict[str, list[tuple[str, str | None]]],
    dict[str, MeasurementTerm],
    dict[str, dict[str, str]],
]:
    """Separate ``mi()`` and ``me()`` syntax from the ordinary response formulas.

    Parameters
    ----------
    specification : MultivariateFormula
        Authored joint formula.

    Returns
    -------
    tuple
        The joint formula with ordinary syntax, ``mi()`` responses by label,
        each response's ``mi()``/``me()`` predictor variables, ``me()`` terms
        by key, and each consumer's map from variable to its source (an
        ``mi()`` response label or an ``me()`` key).

    Raises
    ------
    ValueError
        If an ``mi()`` response uses a family whose missing values cannot be
        latent (`Family.latent_response_obstacle`), residual correlation is requested, ``mi(x)`` names no ``mi()`` response, or a
        variable enters one response through both ``mi()`` and ``me()``.
    """
    responses: dict[str, ResponseModel] = {}
    declared: dict[str, MissingResponse] = {}
    uses: dict[str, list[tuple[str, str | None]]] = {}
    measurements: dict[str, MeasurementTerm] = {}
    for label in specification.response_names:
        rewritten, declaration, used, measured = _rewrite_response(
            label, specification.responses[label]
        )
        responses[label] = rewritten
        if used:
            uses[label] = used
        for variable, noise, group in measured:
            key = f"{label}__{_qvr_name(variable)}"
            term = MeasurementTerm(
                key=key,
                consumer=label,
                variable=variable,
                noise=noise,
                group=group,
                label="me" + "".join(c for c in variable + noise if c.isalnum()),
                latent="me" + "".join(c for c in variable if c.isalnum()),
            )
            if measurements.setdefault(key, term) != term:
                raise ValueError(
                    f"response {label!r}: me({variable}, ...) appears with two "
                    "different noise or grouping columns"
                )
        if declaration is not None:
            family = rewritten.resolved_family
            obstacle = family.latent_response_obstacle
            if obstacle is not None:
                raise ValueError(
                    f"response {label!r}: mi() is not supported for family "
                    f"{family.name!r}, as in brms: a missing value is a latent "
                    f"coordinate of a continuous response law, and {obstacle}"
                )
            declared[label] = declaration
    variables = {item.variable: label for label, item in declared.items()}
    sources: dict[str, dict[str, str]] = {}
    for label, used in uses.items():
        measured_here = {
            term.variable: key
            for key, term in measurements.items()
            if term.consumer == label
        }
        mapping: dict[str, str] = {}
        for variable, _ in used:
            if variable in measured_here:
                source = measured_here[variable]
            elif variable in variables:
                source = variables[variable]
            else:
                raise ValueError(
                    f"response {label!r}: mi({variable}) requires a response "
                    f"'{variable} | mi() ~ ...' in the same joint formula"
                )
            if mapping.setdefault(variable, source) != source:
                raise ValueError(
                    f"response {label!r}: {variable!r} enters through both mi() and me()"
                )
        sources[label] = mapping
    plain = MultivariateFormula(
        responses=responses,
        rescor=specification.rescor,
        response_names=specification.response_names,
        rescor_explicit=True,
    )
    return plain, declared, uses, measurements, sources


def missing_frame(
    data: pd.DataFrame, responses: Mapping[str, MissingResponse]
) -> pd.DataFrame:
    """Give the design evaluators finite placeholders for missing values.

    The placeholders shape design matrices only. Every likelihood and every
    predictor use of these values is replaced by recorded or latent
    coordinates before compilation.

    Parameters
    ----------
    data : pandas.DataFrame
        Original rows.
    responses : Mapping[str, MissingResponse]
        ``mi()`` responses.

    Returns
    -------
    pandas.DataFrame
        A copy with missing ``mi()`` values and their errors filled.

    Raises
    ------
    ValueError
        If an ``mi()`` column is absent, nonnumeric, or entirely missing.
    """
    frame = data.copy()
    for item in responses.values():
        for column in (item.variable, item.noise):
            if column is None:
                continue
            if column not in frame:
                raise ValueError(f"mi() column {column!r} is not in the data")
            values = np.asarray(frame[column], dtype=float)
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                raise ValueError(f"mi() column {column!r} has no recorded value")
            frame[column] = np.where(np.isfinite(values), values, observed[0])
    return frame


def _symbolic(column: FixedColumn, rows: int, binding: str) -> SymbolicColumn:
    """Replace one evaluated column by its completed-value expression.

    Parameters
    ----------
    column : FixedColumn
        Evaluated design column.
    rows : int
        Consumer rows.
    binding : str
        Joint binding of the completed column values.

    Returns
    -------
    SymbolicColumn
        The column contract over the consumer's ``Resp`` axis.
    """
    return SymbolicColumn(
        term=column.term,
        name=column.name,
        parameter=ProgramParameter(
            name=column.qvr_name,
            role="data",
            type=tensor_type(REAL, (IndexLiteral(value=rows, sort=NAT),)),
            axes=("Resp",),
            per_row=True,
        ),
        expression=_var(binding),
    )


def symbolize(
    label: str,
    part: _PreparedFormula,
    frame: pd.DataFrame,
    used: list[tuple[str, str | None]],
    sources: Mapping[str, str],
) -> tuple[FormulaToQVRModule, Formula, list[MissingInput]]:
    """Turn the design columns of ``mi()`` predictors into symbolic columns.

    A column depends on an ``mi()`` variable when setting the variable to zero
    changes it. It must be proportional to the product of its ``mi()``
    variables (checked by doubling each), which is how brms enters special
    terms and their interactions; its remaining factor is the column
    evaluated with every ``mi()`` variable equal to one.

    Parameters
    ----------
    label : str
        Consumer response label.
    part : _PreparedFormula
        The consumer's ordinary preparation.
    frame : pandas.DataFrame
        Rows used for preparation, with placeholders.
    used : list[tuple[str, str or None]]
        ``mi()`` variables and index columns of the consumer.
    sources : Mapping[str, str]
        ``mi()`` variable to its response label.

    Returns
    -------
    tuple
        A lens copy and formula with symbolic columns, and the inputs.

    Raises
    ------
    ValueError
        If an ``mi()`` variable enters a column nonlinearly, or a variable is
        used with two different index columns.
    """
    variables = sorted({variable for variable, _ in used})
    indices: dict[str, str | None] = {}
    for variable, index in used:
        if indices.setdefault(variable, index) != index:
            raise ValueError(
                f"response {label!r}: mi({variable}) uses two different index columns"
            )
    rows = part.formula.modifiers.subset
    included = frame if rows is None else frame.iloc[list(rows.positions)]

    def evaluate(
        formula: Formula, parameter: str, prefix: str, values: Mapping[str, float]
    ) -> dict[str, torch.Tensor]:
        """Evaluate design columns with ``mi()`` variables set to constants.

        Parameters
        ----------
        formula : Formula
            One predictor.
        parameter : str
            Its name in the retained design map.
        prefix : str
            Its local input prefix.
        values : Mapping[str, float]
            Constant of each variable.

        Returns
        -------
        dict[str, torch.Tensor]
            Population-level and varying-slope column values by local input
            name.
        """
        return _evaluated_columns(
            formula, part.designs[parameter], included.assign(**dict(values)), prefix
        )

    def dependence(
        predictor: Formula,
        parameter: str,
        prefix: str,
        key: str,
        name: str,
        ones: Mapping[str, torch.Tensor],
    ) -> list[str]:
        """List the ``mi()`` variables one evaluated column is linear in.

        Parameters
        ----------
        predictor : Formula
            Predictor holding the column.
        parameter : str
            Its name in the retained design map.
        prefix : str
            Its local input prefix.
        key : str
            Local input name of the column.
        name : str
            Column label for errors.
        ones : Mapping[str, torch.Tensor]
            Columns evaluated with every variable equal to one.

        Returns
        -------
        list[str]
            Variables the column is proportional to.

        Raises
        ------
        ValueError
            If a variable enters the column nonlinearly.
        """
        depends = []
        for variable in variables:
            zero = evaluate(
                predictor,
                parameter,
                prefix,
                dict.fromkeys(variables, 1.0) | {variable: 0.0},
            )
            if torch.equal(zero[key], ones[key]):
                continue
            double = evaluate(
                predictor,
                parameter,
                prefix,
                dict.fromkeys(variables, 1.0) | {variable: 2.0},
            )
            if not (
                torch.allclose(zero[key], torch.zeros_like(zero[key]))
                and torch.allclose(double[key], 2 * ones[key])
            ):
                raise ValueError(
                    f"response {label!r}: mi({variable}) enters column "
                    f"{name!r} nonlinearly; mi() terms enter the "
                    "predictor linearly, also in interactions"
                )
            depends.append(variable)
        return depends

    lens = copy(part.lens)
    lens._distributional = dict(part.lens._distributional)
    formula = part.formula
    inputs: list[MissingInput] = []
    for parameter, prefix, predictor in part.lens._predictor_items(part.formula):
        ones = evaluate(predictor, parameter, prefix, dict.fromkeys(variables, 1.0))
        replaced = []
        for column in predictor.fixed_columns:
            if not isinstance(column, FixedColumn) or column.is_intercept:
                replaced.append(column)
                continue
            key = prefix + column.qvr_name
            depends = dependence(predictor, parameter, prefix, key, column.name, ones)
            if not depends:
                replaced.append(column)
                continue
            item = MissingInput(
                consumer=label,
                parameter=parameter,
                prefix=prefix,
                column=column.with_(data=ones[key].numpy().copy()),
                sources=tuple(sources[variable] for variable in depends),
                indices=tuple(indices[variable] for variable in depends),
            )
            inputs.append(item)
            replaced.append(
                _symbolic(column, len(predictor.response_values), item.binding)
            )
        # brms (mi(x) | g): the varying slope multiplies the completed values.
        terms = []
        for term in predictor.random_terms:
            slope = term.column
            if not isinstance(slope, FixedColumn) or slope.is_intercept:
                terms.append(term)
                continue
            key = f"{prefix}random_{slope.qvr_name}"
            depends = dependence(predictor, parameter, prefix, key, slope.name, ones)
            if not depends:
                terms.append(term)
                continue
            item = MissingInput(
                consumer=label,
                parameter=parameter,
                prefix=prefix,
                column=slope.with_(data=ones[key].numpy().copy()),
                sources=tuple(sources[variable] for variable in depends),
                indices=tuple(indices[variable] for variable in depends),
                group=term.group,
                slope=term.slope,
            )
            if item.binding not in {existing.binding for existing in inputs}:
                inputs.append(item)
            terms.append(
                term.with_(
                    column=_symbolic(
                        slope, len(predictor.response_values), item.binding
                    )
                )
            )
        symbolic = {
            column.name for column in replaced if isinstance(column, SymbolicColumn)
        }
        for term in terms:
            if (
                isinstance(term.column, SymbolicColumn)
                and term.column.name not in symbolic
            ):
                # brms writes a varying special-term slope into the predictor
                # only beside its population-level coefficient; alone, its
                # Stan program declares the slope and never uses it.
                raise ValueError(
                    f"response {label!r}: the varying slope {term.column.name!r} "
                    f"of ({term.slope} | {term.group}) uses mi() values and needs "
                    "the same term among the population-level terms; brms omits "
                    "such a slope from the predictor without it"
                )
        predictor = predictor.with_(random_terms=tuple(terms))
        updated = predictor.with_(fixed_columns=tuple(replaced))
        if prefix == "":
            formula = updated
        elif parameter in lens._distributional:
            lens._distributional[parameter] = updated
        elif lens._predictor_graph is not None:
            lens._predictor_graph = lens._predictor_graph.with_(
                leaves=dict(lens._predictor_graph.leaves) | {parameter: updated}
            )
    return lens, formula, inputs


def latent_specs(
    plan: MissingPlan,
    data: pd.DataFrame,
    prepared: Mapping[str, _PreparedFormula],
) -> dict[str, LatentResponseSpec]:
    """Separate recorded and latent coordinates of every ``mi()`` response.

    Parameters
    ----------
    plan : MissingPlan
        ``mi()`` declarations.
    data : pandas.DataFrame
        Original rows, with missing values.
    prepared : Mapping[str, _PreparedFormula]
        Ordinary response preparations, retaining included rows.

    Returns
    -------
    dict[str, LatentResponseSpec]
        Recorded/latent plans by response label.
    """
    specs = {}
    for label, item in plan.responses.items():
        selection = prepared[label].formula.modifiers.subset
        rows = selection or RowSelection(
            size=len(data), positions=tuple(range(len(data)))
        )
        frame = data.iloc[list(rows.positions)]
        specs[label] = bind_latent_response(
            item.variable,
            np.asarray(frame[item.variable], dtype=float),
            standard_error=None
            if item.noise is None
            else np.asarray(frame[item.noise], dtype=float),
            rows=rows,
        )
    for key, term in plan.measurements.items():
        selection = prepared[term.consumer].formula.modifiers.subset
        rows = selection or RowSelection(
            size=len(data), positions=tuple(range(len(data)))
        )
        frame = data.iloc[list(rows.positions)]
        specs[key] = bind_latent_response(
            term.variable,
            np.asarray(frame[term.variable], dtype=float),
            standard_error=np.asarray(frame[term.noise], dtype=float),
            rows=rows,
            group=term.group,
            frame=frame if term.group is not None else None,
        )
    return specs


def _source_positions(
    item: MissingInput,
    source: str,
    index: str | None,
    plan: MissingPlan,
    specs: Mapping[str, LatentResponseSpec],
    data: pd.DataFrame,
    consumer_rows: RowSelection,
) -> tuple[int, ...]:
    """Join consumer rows to a source's completed rows.

    Parameters
    ----------
    item : MissingInput
        Consumer column.
    source : str
        Source response label.
    index : str or None
        Consumer index column.
    plan : MissingPlan
        Declarations, including the source's ``index()`` column.
    specs : Mapping[str, LatentResponseSpec]
        Source row plans.
    data : pandas.DataFrame
        Original rows.
    consumer_rows : RowSelection
        Consumer's included rows.

    Returns
    -------
    tuple[int, ...]
        Source positions of each consumer row.

    Raises
    ------
    ValueError
        If ``mi(x, idx)`` is used without an ``index()`` on the source.
    """
    spec = specs[source]
    if index is None:
        return bind_latent_input(spec, consumer_rows).positions
    identity = plan.responses[source].index
    if identity is None:
        raise ValueError(
            f"mi({plan.responses[source].variable}, {index}) requires "
            f"'{plan.responses[source].variable} | mi() + index(...) ~ ...'"
        )

    def labels(column: str, positions: tuple[int, ...]) -> list[str | int]:
        """Read exact identities from one column.

        Parameters
        ----------
        column : str
            Identity column.
        positions : tuple[int, ...]
            Original rows.

        Returns
        -------
        list[str or int]
            Identities as Python integers or strings.
        """
        values = data[column].iloc[list(positions)].tolist()
        return [
            int(value)
            if float(value).is_integer()
            else str(value)
            if isinstance(value, (int, float, np.integer, np.floating))
            else str(value)
            for value in values
        ]

    return bind_latent_input(
        spec,
        consumer_rows,
        source_index=labels(identity, spec.rows.positions),
        target_index=labels(index, consumer_rows.positions),
        index=index,
    ).positions


def _object(statements: list[Statement], name: str, size: int) -> None:
    """Declare one finite axis.

    Parameters
    ----------
    statements : list[Statement]
        Module declarations.
    name : str
        Axis name.
    size : int
        Cardinality.
    """
    FormulaToQVRModule._append_finite_object(statements, name, size)


#: Site of the vector observation of a joint residual event, as
#: `quivers.formulas._joint_compile` names it.
JOINT_SITE = "formula_joint_response"

_LOG_TWO_PI = 1.8378770664093453
_LOG_PI = 1.1447298858494002


def joint_axis(joint: ObserveStep) -> str:
    """Return the row axis of a joint residual observation.

    Parameters
    ----------
    joint : ObserveStep
        The vector observation of the joint residual event.

    Returns
    -------
    str
        Name of its nominal row axis.

    Raises
    ------
    ValueError
        If the observation carries no named row axis.
    """
    if not isinstance(joint.index, TypeName):
        raise ValueError("a joint residual event requires a named row axis")
    return joint.index.name


def _declare_row_inputs(
    parameters: list[QiecValueParameter], response: _ResponseEmission, axis: str
) -> None:
    """Type one response's real row inputs on its nominal row axis.

    Parameters
    ----------
    parameters : list[QiecValueParameter]
        Program value parameters, extended in place.
    response : _ResponseEmission
        The response whose inputs are declared.
    axis : str
        The response's row axis.
    """
    declared = {parameter.name for parameter in parameters}
    for key, value in response.lens.fixed_column_observations(response.formula).items():
        name = response.names.get(key)
        if (
            name is not None
            and name not in declared
            and value.is_floating_point()
            and value.ndim == 1
        ):
            parameters.append(
                QiecValueParameter(
                    name=name,
                    type_expr=QiecTypeApplication(
                        constructor="Tensor",
                        static_arguments=(QiecTypeName(name="Real"),),
                        indices=(
                            QiecShapeIndex(dimensions=(QiecIndexName(name=axis),)),
                        ),
                    ),
                )
            )
            declared.add(name)


def _joint_completion(
    statements: list[Statement],
    inputs: dict[str, torch.Tensor],
    label: str,
    axis: str,
    spec: LatentResponseSpec,
) -> list[ProgramStep]:
    """Complete one ``mi()`` response of a joint residual event.

    The missing values (or, with known measurement error, every true value)
    are latent coordinates under the improper flat law, as brms declares
    ``Ymi`` and ``Yl`` without a prior. The joint residual law scores the
    completed vector. Cost: O(rows).

    Parameters
    ----------
    statements : list[Statement]
        Module declarations, extended with the recorded and missing axes.
    inputs : dict[str, torch.Tensor]
        Program inputs, extended in place.
    label : str
        Response label.
    axis : str
        Row axis of the joint residual event.
    spec : LatentResponseSpec
        Recorded and latent rows of the response.

    Returns
    -------
    list[ProgramStep]
        Steps binding ``mi_<label>__completed`` on the row axis.
    """
    completed = f"mi_{label}__completed"
    seen, missing = f"MiObserved_{label}", f"MiMissing_{label}"
    observed = torch.as_tensor(spec.observed.data.copy())
    rows = len(spec.rows.positions)
    if spec.standard_error is not None:
        _object(statements, seen, len(spec.observed_positions))
        truth = f"mi_{label}__truth"
        inputs[f"mi_{label}__observed"] = observed
        inputs[f"mi_{label}__sd"] = torch.as_tensor(spec.standard_error.data.copy())
        inputs[f"mi_{label}__observed_index"] = torch.as_tensor(
            spec.observed_positions, dtype=torch.long
        )
        return [
            flat_prior_step(truth, lower=None, upper=None, index=TypeName(name=axis)),
            _let(
                f"mi_{label}__truth_observed",
                _factor(
                    (("j", seen),),
                    body=_index(
                        truth, _index(f"mi_{label}__observed_index", _var("j"))
                    ),
                ),
            ),
            ObserveStep(
                vars=(f"mi_{label}__observed",),
                morphism="Normal",
                args=(
                    DrawArgNamed(
                        parameter="loc",
                        value=DrawArgName(text=f"mi_{label}__truth_observed"),
                    ),
                    DrawArgNamed(
                        parameter="scale", value=DrawArgName(text=f"mi_{label}__sd")
                    ),
                ),
                index=TypeName(name=seen),
            ),
            _let(completed, _var(truth)),
        ]
    recorded = torch.zeros(rows, dtype=observed.dtype)
    recorded[list(spec.observed_positions)] = observed
    inputs[f"mi_{label}__recorded"] = recorded
    if not spec.latent_positions:
        return [
            _let(
                completed,
                _factor(
                    (("i", axis),), body=_index(f"mi_{label}__recorded", _var("i"))
                ),
            )
        ]
    _object(statements, missing, len(spec.latent_positions))
    slot = torch.zeros(rows, dtype=torch.long)
    slot[list(spec.latent_positions)] = torch.arange(len(spec.latent_positions))
    mask = torch.zeros(rows, dtype=observed.dtype)
    mask[list(spec.latent_positions)] = 1.0
    inputs[f"mi_{label}__missing_slot"] = slot
    inputs[f"mi_{label}__missing_mask"] = mask
    return [
        flat_prior_step(
            f"mi_{label}__missing",
            lower=None,
            upper=None,
            index=TypeName(name=missing),
        ),
        _let(
            completed,
            _factor(
                (("i", axis),),
                body=_add(
                    _index(f"mi_{label}__recorded", _var("i")),
                    _mul(
                        _index(f"mi_{label}__missing_mask", _var("i")),
                        _index(
                            f"mi_{label}__missing",
                            _index(f"mi_{label}__missing_slot", _var("i")),
                        ),
                    ),
                ),
            ),
        ),
    ]


def _joint_score(joint: ObserveStep, values: tuple[str, ...]) -> list[ProgramStep]:
    """Score completed response vectors under the joint residual law.

    The multivariate Normal or Student density of each row is written by
    forward substitution through the row's lower-triangular scale factor,
    so the factor is never inverted or multiplied out. Cost:
    O(rows * responses^2).

    Parameters
    ----------
    joint : ObserveStep
        The vector observation the score replaces, naming the location, the
        scale factor, the Student degrees of freedom, and the weights.
    values : tuple[str, ...]
        Row vector of each response in component order.

    Returns
    -------
    list[ProgramStep]
        The standardized residuals and one log-density factor.

    Raises
    ------
    ValueError
        If the observation is not a multivariate Normal or Student law with
        named arguments.
    """
    if joint.morphism not in {"MultivariateNormal", "MultivariateStudentT"}:
        raise ValueError("a joint residual event must be Normal or Student")
    arguments: dict[str, LetExprNode] = {}
    for argument in joint.args or ():
        if not isinstance(argument, DrawArgNamed):
            raise ValueError("a joint residual event requires named arguments")
        arguments[argument.parameter] = draw_argument_expression(argument.value)
    row = joint_axis(joint)
    width = len(values)
    at = _var("row")

    def entry(parameter: str, *components: int) -> LetExprNode:
        """Select one row's coordinate of a joint argument.

        Parameters
        ----------
        parameter : str
            ``loc`` or ``scale_tril``.
        *components : int
            Response coordinates.

        Returns
        -------
        LetExprNode
            The indexed expression at the scoped row.
        """
        return LetExprIndex(
            array=arguments[parameter],
            indices=(
                at,
                *(LetExprLiteral(value=index, integral=True) for index in components),
            ),
        )

    steps: list[ProgramStep] = []
    innovations = tuple(f"formula_joint_innovation_{index}" for index in range(width))
    for index, value in enumerate(values):
        residual = _sub(_index(value, at), entry("loc", index))
        for earlier in range(index):
            residual = _sub(
                residual,
                _mul(
                    entry("scale_tril", index, earlier),
                    _index(innovations[earlier], at),
                ),
            )
        steps.append(
            _let(
                innovations[index],
                _factor(
                    (("row", row),),
                    body=_div(residual, entry("scale_tril", index, index)),
                ),
            )
        )
    log_scale = _add(
        *(_call("log", entry("scale_tril", index, index)) for index in range(width))
    )
    squares = _add(
        *(
            _mul(_index(innovation, at), _index(innovation, at))
            for innovation in innovations
        )
    )
    if joint.morphism == "MultivariateNormal":
        density = _sub(
            _sub(LetExprLiteral(value=-0.5 * width * _LOG_TWO_PI), log_scale),
            _mul(LetExprLiteral(value=0.5), squares),
        )
    else:
        nu = arguments["df"]
        half = _mul(
            LetExprLiteral(value=0.5), _add(nu, LetExprLiteral(value=float(width)))
        )
        steps.append(
            _let(
                "formula_joint_normalizer",
                _sub(
                    _sub(
                        _call("lgamma", half),
                        _call("lgamma", _mul(LetExprLiteral(value=0.5), nu)),
                    ),
                    _mul(
                        LetExprLiteral(value=0.5 * width),
                        _add(_call("log", nu), LetExprLiteral(value=_LOG_PI)),
                    ),
                ),
            )
        )
        density = _sub(
            _sub(_var("formula_joint_normalizer"), log_scale),
            _mul(half, _call("log1p", _div(squares, nu))),
        )
    weights = next(
        (
            option.value.value
            for option in joint.options
            if option.key == "weights" and isinstance(option.value, OptionName)
        ),
        None,
    )
    if weights is not None:
        density = _mul(_index(weights, at), density)
    steps.append(
        ScoreStep(
            name="formula_joint_score",
            value=_call("sum", _factor((("row", row),), body=density)),
        )
    )
    return steps


def apply_missing(
    emission: _JointEmission,
    plan: MissingPlan,
    specs: Mapping[str, LatentResponseSpec],
    data: pd.DataFrame,
    *,
    keep_observation: bool = False,
    fresh: Mapping[str, torch.Tensor] | None = None,
    multipliers: Mapping[str, torch.Tensor] | None = None,
    consumer_rows: Mapping[str, RowSelection] | None = None,
    prior: Callable[[str, str, str | None], SampleStep] | None = None,
) -> tuple[_JointEmission, dict[str, torch.Tensor], set[str]]:
    """Rewrite ``mi()`` observations and bind completed predictor values.

    Cost: one gather per ``mi()`` column and argument, O(n) per draw. A
    joint residual event adds O(n * responses^2) for its density factor.

    Parameters
    ----------
    emission : _JointEmission
        Joint emission of the ordinary responses.
    plan : MissingPlan
        ``mi()`` declarations.
    specs : Mapping[str, LatentResponseSpec]
        Recorded/latent row plans of the fit.
    data : pandas.DataFrame
        Original rows, for index joins.
    keep_observation : bool
        Keep each ``mi()`` response's ordinary observation, or the vector
        observation of a joint residual event, so prediction replay reads
        its law on every row; placeholder values are never scored during
        prediction.
    fresh : Mapping[str, torch.Tensor] or None
        Recorded values of each ``mi()`` response on fresh rows. Fresh rows
        use these values directly and draw no latent coordinates.
    multipliers : Mapping[str, torch.Tensor] or None
        Fresh-row values of each symbolic column's non-``mi()`` factor.
    consumer_rows : Mapping[str, RowSelection] or None
        Included rows of each consumer, for the training joins.
    prior : Callable[[str, str, str or None], SampleStep] or None
        Prior draw of one ``me()`` hyperparameter site, given its site, brms
        class (``meanme``, ``sdme``, ``corme``), and coefficient label.
        Required for training emissions with ``me()`` terms.

    Returns
    -------
    tuple
        The rewritten emission, its additional input bindings, and the input
        names it removes.

    Raises
    ------
    ValueError
        If an ``mi()`` response has no ordinary observation.
    """
    program = next(
        item for item in emission.module.statements if isinstance(item, ProgramDecl)
    )
    statements = [
        item
        for item in emission.module.statements
        if not isinstance(item, (ProgramDecl, ExportDecl))
    ]
    steps: list[ProgramStep] = list(program.draws)
    parameters: list[QiecValueParameter] = list(program.value_parameters)
    inputs: dict[str, torch.Tensor] = {}
    removed: set[str] = set()
    returns = list(program.return_vars)
    targets: list[PriorTarget] = []
    if fresh is None and plan.measurements:
        assert prior is not None
        targets = _append_measurements(statements, steps, inputs, plan, specs, prior)
    for key in plan.measurements:
        if fresh is not None:
            inputs[f"me_{key}__values"] = fresh[key]
    joint = next(
        (
            step
            for step in steps
            if isinstance(step, ObserveStep) and step.vars[0] == JOINT_SITE
        ),
        None,
    )
    for label in plan.responses:
        response = emission.responses[label]
        axis = response.objects.get("Resp", response.prefix + "Resp")
        site = response.names[_qvr_name(response.formula.response_name)]
        completed = f"mi_{label}__completed"
        if fresh is not None:
            inputs[completed] = fresh[label]
            continue
        spec = specs[label]
        if joint is not None:
            steps.extend(_joint_completion(statements, inputs, label, axis, spec))
            continue
        observation = next(
            (
                step
                for step in steps
                if isinstance(step, ObserveStep) and step.vars[0] == site
            ),
            None,
        )
        if observation is None:
            raise ValueError(f"mi() response {label!r} has no ordinary observation")
        # The removed observation typed the response's row inputs; declare
        # them on the response's own row axis instead.
        _declare_row_inputs(parameters, response, axis)
        new: list[ProgramStep] = []
        seen, missing = f"MiObserved_{label}", f"MiMissing_{label}"
        observed_values = torch.as_tensor(spec.observed.data.copy())
        if spec.standard_error is not None:
            _object(statements, seen, len(spec.observed_positions))
            truth = f"mi_{label}__truth"
            new.append(
                SampleStep(
                    vars=(truth,),
                    morphism=observation.morphism,
                    args=observation.args,
                    static_args=observation.static_args,
                    index=observation.index,
                )
            )
            new.append(
                _let(
                    f"mi_{label}__truth_observed",
                    _factor(
                        (("j", seen),),
                        body=_index(
                            truth, _index(f"mi_{label}__observed_index", _var("j"))
                        ),
                    ),
                )
            )
            new.append(
                ObserveStep(
                    vars=(f"mi_{label}__observed",),
                    morphism="Normal",
                    args=(
                        DrawArgNamed(
                            parameter="loc",
                            value=DrawArgName(text=f"mi_{label}__truth_observed"),
                        ),
                        DrawArgNamed(
                            parameter="scale", value=DrawArgName(text=f"mi_{label}__sd")
                        ),
                    ),
                    index=TypeName(name=seen),
                )
            )
            new.append(_let(completed, _var(truth)))
            inputs[f"mi_{label}__observed"] = observed_values
            inputs[f"mi_{label}__sd"] = torch.as_tensor(spec.standard_error.data.copy())
            inputs[f"mi_{label}__observed_index"] = torch.as_tensor(
                spec.observed_positions, dtype=torch.long
            )
        elif spec.latent_positions:
            _object(statements, seen, len(spec.observed_positions))
            _object(statements, missing, len(spec.latent_positions))
            zero = f"mi_{label}__zero"
            new.append(
                _let(zero, _factor((("i", axis),), body=LetExprLiteral(value=0.0)))
            )
            observed_args, latent_args = [], []
            for argument in observation.args or ():
                if not isinstance(argument, DrawArgNamed):
                    raise ValueError(
                        f"mi() response {label!r} requires named family arguments"
                    )
                row = f"mi_{label}__{argument.parameter}"
                # The typed binary operation broadcasts scalar arguments to
                # the row axis; row arguments keep their values.
                new.append(
                    _let(
                        row, _add(draw_argument_expression(argument.value), _var(zero))
                    )
                )
                for name, target, index in (
                    (row + "_observed", seen, f"mi_{label}__observed_index"),
                    (row + "_missing", missing, f"mi_{label}__missing_index"),
                ):
                    new.append(
                        _let(
                            name,
                            _factor(
                                (("j", target),),
                                body=_index(row, _index(index, _var("j"))),
                            ),
                        )
                    )
                observed_args.append(
                    argument.with_(value=DrawArgName(text=row + "_observed"))
                )
                latent_args.append(
                    argument.with_(value=DrawArgName(text=row + "_missing"))
                )
            new.append(
                ObserveStep(
                    vars=(f"mi_{label}__observed",),
                    morphism=observation.morphism,
                    args=tuple(observed_args),
                    static_args=observation.static_args,
                    index=TypeName(name=seen),
                )
            )
            new.append(
                SampleStep(
                    vars=(f"mi_{label}__missing",),
                    morphism=observation.morphism,
                    args=tuple(latent_args),
                    static_args=observation.static_args,
                    index=TypeName(name=missing),
                )
            )
            new.append(
                _let(
                    f"mi_{label}__values",
                    _call(
                        "concat",
                        _var(f"mi_{label}__observed"),
                        _var(f"mi_{label}__missing"),
                    ),
                )
            )
            new.append(
                _let(
                    completed,
                    _factor(
                        (("i", axis),),
                        body=_index(
                            f"mi_{label}__values",
                            _index(f"mi_{label}__order", _var("i")),
                        ),
                    ),
                )
            )
            inputs[f"mi_{label}__observed"] = observed_values
            inputs[f"mi_{label}__observed_index"] = torch.as_tensor(
                spec.observed_positions, dtype=torch.long
            )
            inputs[f"mi_{label}__missing_index"] = torch.as_tensor(
                spec.latent_positions, dtype=torch.long
            )
            inputs[f"mi_{label}__order"] = torch.as_tensor(
                spec.full_to_combined, dtype=torch.long
            )
        else:
            # Every value is recorded: the ordinary observation is unchanged.
            steps.append(_let(completed, _var(site)))
            continue
        if keep_observation:
            steps.extend(new)
        else:
            steps = [step for step in steps if step != observation] + new
            removed.add(site)
            returns = [completed if value == site else value for value in returns]
    if joint is not None and fresh is None and plan.responses and not keep_observation:
        # brms scores the completed vectors under the joint residual law
        # (``multi_normal_cholesky_lpdf(Yl | Mu, LSigma)``).
        recorded = emission.fixed_inputs[JOINT_SITE]
        values: dict[str, str] = {}
        for position, (label, response) in enumerate(emission.responses.items()):
            _declare_row_inputs(parameters, response, joint_axis(joint))
            if label in plan.responses:
                values[label] = f"mi_{label}__completed"
            else:
                values[label] = f"mi_joint__{label}"
                inputs[values[label]] = recorded[..., position].clone()
        steps = [step for step in steps if step != joint]
        steps.extend(_joint_score(joint, tuple(values.values())))
        removed.add(JOINT_SITE)
        returns = [f"mi_{label}__completed" for label in plan.responses]
    for item in plan.inputs:
        consumer = emission.responses[item.consumer]
        axis = consumer.objects.get("Resp", consumer.prefix + "Resp")
        body: LetExprNode = _index(f"{item.binding}__factor", _var("i"))
        for position, (source, index) in enumerate(
            zip(item.sources, item.indices, strict=True)
        ):
            gather = f"{item.binding}__source_{position}"
            values = (
                f"me_{source}__values"
                if source in plan.measurements
                else f"mi_{source}__completed"
            )
            body = _mul(body, _index(values, _index(gather, _var("i"))))
            if fresh is not None:
                rows = len(fresh[source])
                inputs[gather] = torch.arange(rows, dtype=torch.long)
            elif source in plan.measurements:
                inputs[gather] = torch.as_tensor(
                    specs[source].full_to_combined, dtype=torch.long
                )
            else:
                assert consumer_rows is not None
                inputs[gather] = torch.as_tensor(
                    _source_positions(
                        item,
                        source,
                        index,
                        plan,
                        specs,
                        data,
                        consumer_rows[item.consumer],
                    ),
                    dtype=torch.long,
                )
        inputs[f"{item.binding}__factor"] = (
            multipliers[item.binding]
            if multipliers is not None
            else torch.as_tensor(item.column.data.copy())
        )
        steps.append(_let(item.binding, _factor((("i", axis),), body=body)))
    export = next(
        item for item in emission.module.statements if isinstance(item, ExportDecl)
    )
    module = order_prior_dependencies(
        emission.module.with_(
            statements=(
                *statements,
                program.with_(
                    draws=tuple(steps),
                    return_vars=tuple(returns),
                    value_parameters=tuple(parameters),
                ),
                export,
            )
        )
    )
    return (
        replace(
            emission,
            module=module,
            prior_targets=(*emission.prior_targets, *targets),
        ),
        inputs,
        removed,
    )


def _append_measurements(
    statements: list[Statement],
    steps: list[ProgramStep],
    inputs: dict[str, torch.Tensor],
    plan: MissingPlan,
    specs: Mapping[str, LatentResponseSpec],
    prior: Callable[[str, str, str | None], SampleStep],
) -> list[PriorTarget]:
    """Emit brms's noncentered latent values of every ``me()`` term.

    For each term, ``Xme = meanme + sdme * zme`` with ``zme`` standard Normal
    on the term's truth axis (rows, or levels of ``gr``), and each recorded
    noisy value is ``Normal(Xme, sdx)``. Terms of one response sharing a
    grouping are correlated through one LKJ Cholesky factor ``Lme`` when
    ``set_mecor`` is true, as brms emits
    ``Xme = meanme + diag(sdme) * Lme * zme``.

    Parameters
    ----------
    statements : list[Statement]
        Module declarations, extended with the truth and term axes.
    steps : list[ProgramStep]
        Program steps, extended.
    inputs : dict[str, torch.Tensor]
        Runtime bindings, extended.
    plan : MissingPlan
        Declarations.
    specs : Mapping[str, LatentResponseSpec]
        Recorded and latent coordinates of every term.
    prior : Callable
        Prior draw of one hyperparameter site.

    Returns
    -------
    list[PriorTarget]
        Overrideable ``meanme``, ``sdme``, and ``corme`` sites.
    """
    blocks: dict[tuple[str, str | None], list[MeasurementTerm]] = {}
    for term in plan.measurements.values():
        blocks.setdefault((term.consumer, term.group), []).append(term)
    targets: list[PriorTarget] = []
    for (consumer, group), terms in blocks.items():
        name = _qvr_name(f"{consumer}_{group or 'rows'}")
        truth = f"MeTruth_{name}"
        size = specs[terms[0].key].truth_size
        _object(statements, truth, size)
        for term in terms:
            mean, scale = f"me_{term.key}__mean", f"me_{term.key}__sd"
            steps.append(prior(mean, "meanme", term.latent))
            steps.append(prior(scale, "sdme", term.latent))
            steps.append(
                SampleStep(
                    vars=(f"me_{term.key}__z",),
                    morphism="Normal",
                    args=(
                        DrawArgNamed(parameter="loc", value=DrawArgScalar(value=0.0)),
                        DrawArgNamed(parameter="scale", value=DrawArgScalar(value=1.0)),
                    ),
                    index=TypeName(name=truth),
                )
            )
            targets.append(
                PriorTarget(
                    site=mean, class_="meanme", coef=term.latent, response=consumer
                )
            )
            targets.append(
                PriorTarget(
                    site=scale,
                    class_="sdme",
                    coef=term.latent,
                    response=consumer,
                    lower=0.0,
                )
            )
        correlated = plan.mecor.get(consumer, True) and len(terms) > 1
        factor = f"me_{name}__cor"
        if correlated:
            axis = f"MeTerms_{name}"
            _object(statements, axis, len(terms))
            steps.append(prior(factor, "corme", None).with_(index=TypeName(name=axis)))
            targets.append(
                PriorTarget(
                    site=factor,
                    class_="corme",
                    response=consumer,
                    event_rank=2,
                    component_labels=tuple(term.latent for term in terms),
                )
            )
        for k, term in enumerate(terms):
            innovation: LetExprNode = _index(f"me_{term.key}__z", _var("u"))
            if correlated:
                innovation = _add(
                    *(
                        _mul(
                            _index(
                                factor,
                                LetExprLiteral(value=k, integral=True),
                                LetExprLiteral(value=j, integral=True),
                            ),
                            _index(f"me_{terms[j].key}__z", _var("u")),
                        )
                        for j in range(k + 1)
                    )
                )
            values = f"me_{term.key}__values"
            steps.append(
                _let(
                    values,
                    _factor(
                        (("u", truth),),
                        body=_add(
                            _var(f"me_{term.key}__mean"),
                            _mul(_var(f"me_{term.key}__sd"), innovation),
                        ),
                    ),
                )
            )
            spec = specs[term.key]
            observed = f"MeObserved_{_qvr_name(term.key)}"
            _object(statements, observed, len(spec.observed_positions))
            steps.append(
                _let(
                    f"me_{term.key}__values_observed",
                    _factor(
                        (("j", observed),),
                        body=_index(
                            values, _index(f"me_{term.key}__observed_index", _var("j"))
                        ),
                    ),
                )
            )
            steps.append(
                ObserveStep(
                    vars=(f"me_{term.key}__noisy",),
                    morphism="Normal",
                    args=(
                        DrawArgNamed(
                            parameter="loc",
                            value=DrawArgName(text=f"me_{term.key}__values_observed"),
                        ),
                        DrawArgNamed(
                            parameter="scale",
                            value=DrawArgName(text=f"me_{term.key}__noise"),
                        ),
                    ),
                    index=TypeName(name=observed),
                )
            )
            assert spec.standard_error is not None
            inputs[f"me_{term.key}__noisy"] = torch.as_tensor(spec.observed.data.copy())
            inputs[f"me_{term.key}__noise"] = torch.as_tensor(
                spec.standard_error.data.copy()
            )
            inputs[f"me_{term.key}__observed_index"] = torch.as_tensor(
                spec.observed_positions, dtype=torch.long
            )
    return targets


def missing_multipliers(
    plan: MissingPlan,
    designs: Mapping[str, _PreparedFormula],
    frame: pd.DataFrame,
) -> dict[str, torch.Tensor]:
    """Evaluate each symbolic column's non-``mi()`` factor on fresh rows.

    Parameters
    ----------
    plan : MissingPlan
        ``mi()`` declarations.
    designs : Mapping[str, _PreparedFormula]
        Response preparations with the fitted design evaluators.
    frame : pandas.DataFrame
        Fresh rows.

    Returns
    -------
    dict[str, torch.Tensor]
        Factor values by binding.
    """
    variables = {
        plan.measurements[source].variable
        if source in plan.measurements
        else plan.responses[source].variable
        for item in plan.inputs
        for source in item.sources
    }
    ones = frame.assign(**dict.fromkeys(variables, 1.0))
    result = {}
    for item in plan.inputs:
        part = designs[item.consumer]
        column = item.column
        predictor = (
            Formula(
                formula="",
                response_name="",
                fixed_columns=(column,),
                response_values=np.zeros(len(frame)),
            )
            if item.group is None
            else Formula(
                formula="",
                response_name="",
                random_terms=(
                    RandomTerm(slope=item.slope, group=item.group, column=column),
                ),
                response_values=np.zeros(len(frame)),
            )
        )
        values = _evaluated_columns(
            predictor, part.designs[item.parameter], ones, item.prefix
        )
        result[item.binding] = values[item.input_name]
    return result


def _evaluated_columns(
    formula: Formula, design: DesignMatrices, frame: pd.DataFrame, prefix: str
) -> dict[str, torch.Tensor]:
    """Evaluate the row data of one predictor.

    Parameters
    ----------
    formula : Formula
        Predictor.
    design : DesignMatrices
        Its retained evaluators.
    frame : pandas.DataFrame
        Rows.
    prefix : str
        Its local input prefix.

    Returns
    -------
    dict[str, torch.Tensor]
        Population-level and varying-slope column values by local input name.
    """
    rows = RowSource(frame=frame, designs={prefix: design}).predictor(prefix)
    return finite_inputs(
        bind_inputs(
            item
            for term in data_terms(formula, prefix)
            for item in term.at(rows).inputs()
        )
    )


def fresh_values(plan: MissingPlan, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Read recorded ``mi()`` values on fresh rows.

    brms returns missing predictions for fresh rows whose ``mi()`` values
    are missing; such rows are refused here with that reason.

    Parameters
    ----------
    plan : MissingPlan
        ``mi()`` declarations.
    frame : pandas.DataFrame
        Fresh rows.

    Returns
    -------
    dict[str, torch.Tensor]
        Values by response label.

    Raises
    ------
    ValueError
        If a column is absent or has missing values.
    """
    result = {}
    columns = {label: item.variable for label, item in plan.responses.items()} | {
        key: term.variable for key, term in plan.measurements.items()
    }
    for label, variable in columns.items():
        item = MissingResponse(label=label, variable=variable)
        if item.variable not in frame:
            raise ValueError(f"new_data must contain the mi() column {item.variable!r}")
        values = np.asarray(frame[item.variable], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"new_data has missing values of the mi() column {item.variable!r}; "
                "brms returns missing predictions for those rows, so supply "
                "complete rows"
            )
        result[label] = torch.as_tensor(values.copy())
    return result
