"""Plan the linear predictors of a model as sums of terms.

A predictor names a value over the rows and, for a vector-valued parameter, an
output axis. The location, a distributional parameter and a nonlinear parameter
are one thing here: a namespace, a link, and a tuple of terms.
`plan_predictors` is the only function that reads the parsed terms of a
`Formula` by kind; every other reader asks the terms.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import ProgramStep
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _index, _let, _var
from quivers.formulas._brms_priors import DefaultPriorPolicy, sdgp_prior
from quivers.formulas._category_formulas import is_scale_parameter
from quivers.formulas._design import SymbolicColumn
from quivers.formulas._gp import process_term
from quivers.formulas._group_law import GroupBlockLaw
from quivers.formulas._group_plan import (
    GroupTerm,
    coefficient_binding,
    coefficient_owner,
    row_map,
)
from quivers.formulas._field_term import field_terms
from quivers.formulas._monotonic_emission import monotonic_rows
from quivers.formulas._population_coordinates import (
    PopulationCoordinates,
    PopulationTerm,
    population_coordinates,
)
from quivers.formulas._predictor_graph import PredictorGraph
from quivers.formulas._row_source import PredictorRows, RowSource, fresh_columns
from quivers.formulas._smooth import BasisTerm
from quivers.formulas._variable_catalogue import brms_rename
from quivers.formulas._term import (
    AxisDecl,
    EffectSource,
    FormulaPart,
    PopulationCoefficient,
    VariableDecl,
    SmoothPart,
    ColumnName,
    Contribution,
    flat_index,
    over_output,
    Emission,
    InputDecl,
    OutputAxis,
    ProgramNames,
    SiteDecl,
    Term,
    column_rms,
)
from quivers.formulas.family import Family, Link, links
from quivers.formulas.formula import FixedColumn, Formula, RandomTerm

if TYPE_CHECKING:
    from quivers.formulas._name_scope import NameScope

__all__: list[str] = []


class OffsetTerm(ValueModel):
    """The offsets of one predictor.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    columns : tuple[FixedColumn, ...]
        Evaluated offsets, in authored order.
    output : OutputAxis or None
        Axis the offsets are repeated along.
    """

    prefix: str
    columns: tuple[FixedColumn, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    output: OutputAxis | None = None

    @property
    def label(self) -> str:
        """Return the authored offsets.

        Returns
        -------
        str
            Their labels, joined by plus signs.
        """
        return " + ".join(column.name for column in self.columns)

    def at(self, rows: PredictorRows) -> OffsetTerm:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        OffsetTerm
            The offsets evaluated on the rows.
        """
        if rows.frame is None:
            return self
        return self.with_(columns=fresh_columns(self.columns, rows.column))

    def column_names(self) -> tuple[ColumnName, ...]:
        """List the data columns the term reads under their own names.

        Returns
        -------
        tuple[ColumnName, ...]
            Empty: the inputs are named after the term.
        """
        return ()

    def variables(self, scope: NameScope) -> tuple[VariableDecl, ...]:
        """Name the term's draws as brms does.

        Parameters
        ----------
        scope : NameScope
            Names of the term's predictor.

        Returns
        -------
        tuple[VariableDecl, ...]
            Empty: offsets have no draws.
        """
        return ()

    def coefficients(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the coefficients of the term for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        torch.Tensor
            Never: the term has no coefficients.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(f"{self.label!r} has no coefficients")

    def population(self) -> tuple[PopulationCoefficient, ...]:
        """List the term's population-level coefficients.

        Returns
        -------
        tuple[PopulationCoefficient, ...]
            Empty.
        """
        return ()

    def smooth_part(self, label: str) -> SmoothPart | None:
        """Return the term's share of one smooth or Gaussian process.

        Parameters
        ----------
        label : str
            Authored smooth term.

        Returns
        -------
        SmoothPart or None
            None.
        """
        return None

    def effects(self) -> tuple[EffectSource, ...]:
        """List the design terms whose variables brms plots.

        Returns
        -------
        tuple[EffectSource, ...]
            Each offset.
        """
        return tuple(
            EffectSource(label=column.term, kind="fe") for column in self.columns
        )

    def contributions(self) -> tuple[Contribution, ...]:
        """List the bound summands of the term.

        Returns
        -------
        tuple[Contribution, ...]
            Each offset, shared by every output coordinate.
        """
        return tuple(
            Contribution(
                name=name,
                rows=True,
                coordinates=False,
                part=FormulaPart(field="offsets", key=str(index)),
            )
            for index, name in enumerate(self.names())
        )

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            Empty.
        """
        return ()

    def names(self) -> tuple[str, ...]:
        """Name the input of each offset.

        Returns
        -------
        tuple[str, ...]
            Input names in authored order.
        """
        return tuple(
            f"{self.prefix}formula_offset_{index}"
            for index, _ in enumerate(self.columns)
        )

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the offsets.

        Returns
        -------
        tuple[InputDecl, ...]
            One inferred row vector per offset.
        """
        return tuple(
            InputDecl(
                name=name,
                element="Real",
                axes=("Resp",),
                declared=False,
                values=torch.as_tensor(column.data.copy(), dtype=torch.float32),
            )
            for name, column in zip(self.names(), self.columns, strict=True)
        )

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            Empty.
        """
        return ()

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            None.
        """
        return None

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Draw the term's sites and bind its contributions.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            Empty.
        """
        return ()


class HostTerm(ValueModel):
    """A contribution computed by a host module at every evaluation.

    Parameters
    ----------
    name : str
        Input holding the contribution: one value per row, or one per row and
        coordinate, rows outermost, for a term with an output axis.
    output : OutputAxis or None
        Axis of the coordinates.
    """

    name: str
    output: OutputAxis | None = None

    @property
    def label(self) -> str:
        """Return the host input.

        Returns
        -------
        str
            Input name.
        """
        return self.name

    def at(self, rows: PredictorRows) -> HostTerm:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        HostTerm
            The term itself: the host evaluates its own rows.
        """
        return self

    def column_names(self) -> tuple[ColumnName, ...]:
        """List the data columns the term reads under their own names.

        Returns
        -------
        tuple[ColumnName, ...]
            Empty: the inputs are named after the term.
        """
        return ()

    def variables(self, scope: NameScope) -> tuple[VariableDecl, ...]:
        """Name the term's draws as brms does.

        Parameters
        ----------
        scope : NameScope
            Names of the term's predictor.

        Returns
        -------
        tuple[VariableDecl, ...]
            Empty: the host has no draws.
        """
        return ()

    def coefficients(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the coefficients of the term for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        torch.Tensor
            Never: the term has no coefficients.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(f"{self.label!r} has no coefficients")

    def population(self) -> tuple[PopulationCoefficient, ...]:
        """List the term's population-level coefficients.

        Returns
        -------
        tuple[PopulationCoefficient, ...]
            Empty.
        """
        return ()

    def smooth_part(self, label: str) -> SmoothPart | None:
        """Return the term's share of one smooth or Gaussian process.

        Parameters
        ----------
        label : str
            Authored smooth term.

        Returns
        -------
        SmoothPart or None
            None.
        """
        return None

    def effects(self) -> tuple[EffectSource, ...]:
        """List the design terms whose variables brms plots.

        Returns
        -------
        tuple[EffectSource, ...]
            Empty: the host reads its own data.
        """
        return ()

    def contributions(self) -> tuple[Contribution, ...]:
        """List the bound summands of the term.

        Returns
        -------
        tuple[Contribution, ...]
            The host input, by coordinate for a term with an output axis.
        """
        return (
            Contribution(
                name=self.name if self.output is None else self.name + "_by_coordinate",
                rows=True,
                coordinates=self.output is not None,
            ),
        )

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            Empty.
        """
        return ()

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs of the design.

        Returns
        -------
        tuple[InputDecl, ...]
            Empty: the host binds its own value at every evaluation.
        """
        return ()

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            Empty.
        """
        return ()

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            None.
        """
        return None

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Bind the host values by coordinate.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            Empty for a scalar term.
        """
        if self.output is None:
            return ()
        value = _index(self.name, flat_index(self.output, _var("response")))
        return (
            _let(
                self.name + "_by_coordinate",
                over_output(self.output, value, ("response", "Resp")),
            ),
        )


class PredictorOptions(ValueModel):
    """Configuration shared by the predictors of one model.

    Parameters
    ----------
    coefficient_prior : str
        Default prior of a population coefficient.
    scale_prior : str
        Default prior of a standard deviation.
    correlation_prior : str
        Default prior of a correlation factor.
    default_priors : {"quivers", "brms"}
        Default prior policy.
    center : bool or None
        Sample the intercept of the column-centered design; None follows the
        policy.
    decomp : {"none", "QR"}
        Coordinates of the location coefficients.
    coordinates : bool
        Whether the family admits centered and QR coordinates.
    host : str or None
        Input holding a host contribution to the location.
    names : ProgramNames
        Names of the columns and contributions in the program.
    """

    coefficient_prior: str
    scale_prior: str
    correlation_prior: str
    default_priors: DefaultPriorPolicy
    center: bool | None
    decomp: Literal["none", "QR"]
    coordinates: bool
    host: str | None
    names: ProgramNames = dx.field(
        default_factory=ProgramNames, opaque=True, extras=OPAQUE_BY_VALUE
    )


class Predictor(ValueModel):
    """One linear predictor: a namespace, a link, and a sum of terms.

    Parameters
    ----------
    parameter : str
        Parameter the predictor feeds.
    role : {"dpar", "nlpar"}
        Whether it is a parameter of the response law or of a nonlinear
        expression. Names only.
    prefix : str
        Namespace of its coefficients and inputs.
    link : Link
        Link of the parameter it feeds; the identity for a nonlinear
        parameter.
    output : OutputAxis or None
        Coordinates of a vector-valued parameter.
    terms : tuple[Term, ...]
        Terms in emission order.
    nonlinear : bool
        Whether a nonlinear expression over the rows gives the value.
    """

    parameter: str
    role: Literal["dpar", "nlpar"]
    prefix: str
    link: Link = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    output: OutputAxis | None = None
    terms: tuple[Term, ...] = dx.field(default=(), opaque=True, extras=OPAQUE_BY_VALUE)
    nonlinear: bool = False

    def at(self, rows: RowSource) -> Predictor:
        """Return the predictor on other rows.

        Parameters
        ----------
        rows : RowSource
            Rows of one request.

        Returns
        -------
        Predictor
            The predictor with every term on the rows.
        """
        read = rows.predictor(self.prefix)
        return self.with_(terms=tuple(term.at(read) for term in self.terms))

    def part(self, output: OutputAxis | None) -> tuple[Term, ...]:
        """Select the terms over one output axis.

        Parameters
        ----------
        output : OutputAxis or None
            Axis of the terms; None for scalar terms.

        Returns
        -------
        tuple[Term, ...]
            Terms in emission order.
        """
        return tuple(term for term in self.terms if term.output == output)

    def parts(self) -> tuple[tuple[OutputAxis | None, tuple[Term, ...]], ...]:
        """Group the terms by output axis.

        Returns
        -------
        tuple[tuple[OutputAxis or None, tuple[Term, ...]], ...]
            Each axis with its terms, in order of first appearance.
        """
        axes = list(dict.fromkeys(term.output for term in self.terms))
        return tuple((axis, self.part(axis)) for axis in axes)

    @property
    def rowwise(self) -> bool:
        """Report whether the predictor's own part varies over the rows.

        Returns
        -------
        bool
            True when a term of the part over the predictor's output axis has
            a row-varying design, or a nonlinear expression gives the value.
        """
        return self.nonlinear or any(
            contribution.rows
            for term in self.part(self.output)
            for contribution in term.contributions()
        )

    @property
    def scope(self) -> dict[str, str]:
        """Return the prior-target coordinates of the predictor's sites.

        Returns
        -------
        dict[str, str]
            The parameter under ``dpar`` or ``nlpar``.
        """
        return {self.role: self.parameter}


def response_axis(family: Family, labels: tuple[str, ...]) -> OutputAxis | None:
    """Return the axis of the free coordinates of a vector-valued location.

    Parameters
    ----------
    family : Family
        Response family.
    labels : tuple[str, ...]
        Every category or component label, in retained order.

    Returns
    -------
    OutputAxis or None
        The free coordinates, or None for a scalar location.
    """
    if family.response is None:
        return None
    free = family.response.coordinate_indices(labels)
    return OutputAxis(
        name="FormulaCategoryLogit",
        size=len(free),
        labels=tuple(str(labels[index]) for index in free),
        parameters=tuple("mu" + brms_rename(str(labels[index])) for index in free),
    )


def transition_axis(family: Family, categories: int) -> OutputAxis:
    """Return the axis of the transitions of an ordinal response.

    Parameters
    ----------
    family : Family
        Family with ordered categories.
    categories : int
        Number of response categories.

    Returns
    -------
    OutputAxis
        One coordinate per transition, named by position.

    Raises
    ------
    ValueError
        If the family has no ordered categories.
    """
    ordinal = family.response_ordinal()
    if ordinal is None:
        raise ValueError("cs terms require an ordinal response family")
    return OutputAxis(
        name="FormulaCutpoint",
        size=categories - 1 - int(ordinal.kind == "hurdle_cumulative"),
    )


def _coordinates(
    formula: Formula,
    prefix: str,
    role: Literal["dpar", "nlpar"],
    columns: tuple[FixedColumn | SymbolicColumn, ...],
    options: PredictorOptions,
) -> PopulationCoordinates | None:
    """Decide the sampled coordinates of one predictor's population effects.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace, empty for the location.
    role : {"dpar", "nlpar"}
        Role of the predictor.
    columns : tuple[FixedColumn or SymbolicColumn, ...]
        Scalar columns of the predictor.
    options : PredictorOptions
        Centering and decomposition options.

    Returns
    -------
    PopulationCoordinates or None
        The coordinate plan, or None for the coefficients themselves.
    """
    if not options.coordinates or role == "nlpar":
        return None
    center = options.default_priors == "brms"
    if prefix:
        center = center or options.center is True
    elif options.center is not None:
        center = options.center
    return population_coordinates(
        formula.with_(fixed_columns=columns),
        prefix,
        center=center,
        decomp="none" if prefix else options.decomp,
    )


def design_terms(
    formula: Formula,
    prefix: str,
    *,
    output: OutputAxis | None = None,
    scale_prior: str = "",
    process_prior: Callable[[], str] = str,
) -> tuple[Term, ...]:
    """Build the coupled design blocks of one predictor.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace.
    output : OutputAxis or None
        Axis of the predictor.
    scale_prior : str
        Default prior of a smooth's scale; empty when only the inputs of the
        terms are read.
    process_prior : Callable[[], str]
        Default prior of a Gaussian process's scale, as for ``scale_prior``.

    Returns
    -------
    tuple[Term, ...]
        One term per block.
    """
    terms: list[Term] = []
    scales: set[str] = set()
    for block in formula.design_terms:
        name = prefix + block.qvr_name
        if block.gp is not None:
            terms.append(
                process_term(
                    label=block.term,
                    name=name,
                    prefix=prefix,
                    block=block.gp,
                    matrix=block.data,
                    columns=block.column_indices,
                    scale_prior=process_prior(),
                    output=output,
                )
            )
            continue
        scale = prefix + (block.scale_group or block.qvr_name) + "_scale"
        terms.append(
            BasisTerm(
                label=block.term,
                name=name,
                prefix=prefix,
                matrix=block.data,
                columns=block.column_indices,
                scale=scale,
                owns_scale=scale not in scales,
                prior=scale_prior,
                output=output,
            )
        )
        scales.add(scale)
    return tuple(terms)


def _slope_position(formula: Formula, term: RandomTerm) -> tuple[int, int]:
    """Locate a slope among the design columns of its term.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    term : RandomTerm
        Group-level coefficient.

    Returns
    -------
    tuple[int, int]
        Position of the slope's column, and the number of columns. A slope
        shared by several grouping factors has one column.
    """
    if term.column is None:
        return 0, 1
    names = list(
        dict.fromkeys(
            other.column.name
            for other in formula.random_terms
            if other.column is not None and other.column.term == term.column.term
        )
    )
    return names.index(term.column.name), len(names)


def group_term(
    formula: Formula,
    prefix: str,
    term: RandomTerm,
    output: OutputAxis | None,
    laws: Mapping[tuple[str, str, str], GroupBlockLaw],
    names: ProgramNames = ProgramNames(),
) -> GroupTerm:
    """Build one group-level coefficient of a predictor.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace.
    term : RandomTerm
        Parsed coefficient.
    output : OutputAxis or None
        Axis of its coefficient vectors.
    laws : Mapping[tuple[str, str, str], GroupBlockLaw]
        Laws of the blocks the program emits.

    names : ProgramNames
        Names of the contributions in the program.

    Returns
    -------
    GroupTerm
        The planned term.

    Raises
    ------
    ValueError
        If a multi-membership coefficient has no retained covariate.
    """
    membership = formula.grouping.membership(term.group)
    covariate = formula.grouping.covariate(term.group, term.slope)
    if membership is not None and covariate is None:
        raise ValueError(
            f"multi-membership coefficient {term.slope!r} has no member covariate"
        )
    position, width = _slope_position(formula, term)
    return GroupTerm(
        prefix=prefix,
        binding=names.binding(
            coefficient_owner(prefix, term.group, term.slope),
            coefficient_binding(prefix, term.group, term.slope) + "_per_row",
        ),
        group=term.group,
        slope=term.slope,
        rows=row_map(formula, term, prefix, vector=output is not None),
        column=term.column,
        membership=membership,
        covariate=covariate if membership is not None else None,
        law=laws.get((prefix, term.group, term.slope)),
        position=position,
        width=width,
        output=output,
    )


def _population(
    columns: tuple[FixedColumn | SymbolicColumn, ...],
    formula: Formula,
    prefix: str,
    role: Literal["dpar", "nlpar"],
    output: OutputAxis | None,
    options: PredictorOptions,
    *,
    transform: bool,
) -> tuple[Term, ...]:
    """Build the population block of one predictor over one axis.

    Parameters
    ----------
    columns : tuple[FixedColumn or SymbolicColumn, ...]
        Design columns, in authored order.
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace.
    role : {"dpar", "nlpar"}
        Role of the predictor.
    output : OutputAxis or None
        Axis of the coefficient vectors.
    options : PredictorOptions
        Default priors and coordinates.
    transform : bool
        Whether the block may take centered or QR coordinates.

    Returns
    -------
    tuple[Term, ...]
        The block, or nothing without columns.
    """
    coordinates = (
        _coordinates(formula, prefix, role, columns, options) if transform else None
    )
    if not columns:
        return ()
    return (
        PopulationTerm(
            prefix=prefix,
            names=options.names,
            prior=options.coefficient_prior,
            columns=columns,
            scales=tuple(
                column_rms(column.data)
                if isinstance(column, FixedColumn) and not column.is_intercept
                else None
                for column in columns
            ),
            coordinates=coordinates,
            output=output,
        ),
    )


def predictor_terms(
    formula: Formula,
    prefix: str,
    role: Literal["dpar", "nlpar"],
    *,
    process_prior: Callable[[], str],
    options: PredictorOptions,
    output: OutputAxis | None,
    transitions: OutputAxis | None,
    laws: Mapping[tuple[str, str, str], GroupBlockLaw],
) -> tuple[Term, ...]:
    """Build the terms of one predictor, in emission order.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace, empty for the location.
    role : {"dpar", "nlpar"}
        Role of the predictor.
    process_prior : Callable[[], str]
        Default prior of a Gaussian process's scale.
    options : PredictorOptions
        Default priors, coordinates and the host contribution.
    output : OutputAxis or None
        Axis of the predictor's coefficient vectors.
    transitions : OutputAxis or None
        Axis of category-specific coefficients.
    laws : Mapping[tuple[str, str, str], GroupBlockLaw]
        Laws of the group blocks the program emits, by coefficient.

    Returns
    -------
    tuple[Term, ...]
        Every term of the predictor.

    Raises
    ------
    ValueError
        If a category-specific coefficient has no transition axis, or a
        symbolic column belongs to other rows.
    """
    rows = len(formula.response_values)
    for column in (
        *formula.fixed_columns,
        *(term.column for term in formula.random_terms),
    ):
        if isinstance(column, SymbolicColumn):
            column.validate_rows("Resp", rows)
    specific_columns = tuple(
        column for column in formula.fixed_columns if column.category_specific
    )
    specific_randoms = tuple(
        term
        for term in formula.random_terms
        if term.column is not None and term.column.category_specific
    )
    if (specific_columns or specific_randoms) and transitions is None:
        raise ValueError("cs terms require an ordinal response family")
    offsets: tuple[Term, ...] = (
        (OffsetTerm(prefix=prefix, columns=formula.offsets, output=output),)
        if formula.offsets
        else ()
    )
    design = design_terms(
        formula,
        prefix,
        output=output,
        scale_prior=options.scale_prior,
        process_prior=process_prior,
    )
    special: tuple[Term, ...] = monotonic_rows(
        formula, prefix, prior=options.coefficient_prior, output=output
    )
    population = _population(
        tuple(
            column for column in formula.fixed_columns if not column.category_specific
        ),
        formula,
        prefix,
        role,
        output,
        options,
        transform=output is None,
    )
    host: tuple[Term, ...] = (
        (HostTerm(name=options.host, output=output),)
        if options.host is not None and not prefix
        else ()
    )
    groups: tuple[Term, ...] = tuple(
        group_term(formula, prefix, term, output, laws, options.names)
        for term in formula.random_terms
        if term not in specific_randoms
    )
    fields: tuple[Term, ...] = field_terms(
        formula,
        scale_prior=options.scale_prior,
        correlation_prior=options.correlation_prior,
    )
    specific: tuple[Term, ...] = (
        *_population(
            specific_columns,
            formula,
            prefix,
            role,
            transitions,
            options,
            transform=False,
        ),
        *(
            group_term(formula, prefix, term, transitions, laws, options.names)
            for term in specific_randoms
        ),
    )
    return (
        *offsets,
        *design,
        *special,
        *population,
        *host,
        *groups,
        *fields,
        *specific,
    )


def data_terms(formula: Formula, prefix: str) -> tuple[Term, ...]:
    """Build the terms of one predictor as readers of row data.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace.

    Returns
    -------
    tuple[Term, ...]
        The terms without priors, coordinates, or group laws. The slope of a
        multi-membership coefficient is its row column.
    """
    grouping = formula.grouping.with_(memberships=())
    return predictor_terms(
        formula.with_(grouping=grouping),
        prefix,
        "dpar",
        process_prior=str,
        options=PredictorOptions(
            coefficient_prior="",
            scale_prior="",
            correlation_prior="",
            default_priors="quivers",
            center=None,
            decomp="none",
            coordinates=False,
            host=None,
        ),
        output=None,
        transitions=OutputAxis(name="FormulaCutpoint", size=1),
        laws={},
    )


def slot_link(family: Family, parameter: str) -> Link:
    """Return the link of one parameter of a response law.

    Parameters
    ----------
    family : Family
        Response family.
    parameter : str
        Parameter name; a category's own location or scale takes the link of
        the vector-valued parameter it belongs to.

    Returns
    -------
    Link
        The declared link.
    """
    if parameter == "mu":
        return family.location_link
    for auxiliary in family.aux_params:
        if auxiliary.name == parameter:
            return auxiliary.link
    scale = family.component_scale
    if scale is not None and is_scale_parameter(family, parameter):
        return scale.link
    return links["identity"]


def plan_predictors(
    formula: Formula,
    family: Family,
    *,
    distributional: Mapping[str, Formula],
    graph: PredictorGraph | None,
    options: PredictorOptions,
    outputs: Mapping[str, OutputAxis],
    transitions: OutputAxis | None,
    laws: Mapping[tuple[str, str, str], GroupBlockLaw],
) -> tuple[Predictor, ...]:
    """Plan every linear predictor of a model.

    Parameters
    ----------
    formula : Formula
        Location predictor and response.
    family : Family
        Response family.
    distributional : Mapping[str, Formula]
        Predictors of other parameters of the response law.
    graph : PredictorGraph or None
        Nonlinear parameters and expressions.
    options : PredictorOptions
        Shared configuration.
    outputs : Mapping[str, OutputAxis]
        Axis of each vector-valued predictor, by namespace.
    transitions : OutputAxis or None
        Axis of category-specific coefficients.
    laws : Mapping[tuple[str, str, str], GroupBlockLaw]
        Laws of the group blocks the program emits, by coefficient.

    Returns
    -------
    tuple[Predictor, ...]
        The location, the distributional parameters, the nonlinear parameters,
        and the parameters only a nonlinear expression gives.
    """
    roots = {} if graph is None else graph.roots
    entries: list[tuple[str, Literal["dpar", "nlpar"], str, Formula]] = [
        ("mu", "dpar", "", formula),
        *(
            (name, "dpar", f"dpar_{name}_", value)
            for name, value in distributional.items()
        ),
        *(
            (name, "nlpar", f"nlpar_{name}_", value)
            for name, value in ({} if graph is None else graph.leaves).items()
        ),
    ]
    predictors: list[Predictor] = []
    for name, role, prefix, value in entries:
        output = outputs.get(prefix)
        predictors.append(
            Predictor(
                parameter=name,
                role=role,
                prefix=prefix,
                link=links["identity"] if role == "nlpar" else slot_link(family, name),
                output=output,
                terms=predictor_terms(
                    value,
                    prefix,
                    role,
                    process_prior=partial(sdgp_prior, value.response_values, family),
                    options=options,
                    output=output,
                    transitions=transitions,
                    laws=laws,
                ),
                nonlinear=role == "dpar" and name in roots,
            )
        )
    planned = {item.parameter for item in predictors if item.role == "dpar"}
    predictors.extend(
        Predictor(
            parameter=name,
            role="dpar",
            prefix=f"dpar_{name}_",
            link=slot_link(family, name),
            nonlinear=True,
        )
        for name in roots
        if name not in planned
    )
    return tuple(predictors)
