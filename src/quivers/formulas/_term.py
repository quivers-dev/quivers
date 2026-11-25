"""Declarations of the terms of a linear predictor.

Every additive term of a predictor is a row design times a block of
coefficients,

    contribution[n, k] = sum_j D[n, j] theta[j, k],

where ``k`` ranges over the output axis of the predictor and is absent for a
scalar predictor. A population effect's design is a column, a group effect's an
index with a covariate, a smooth's its basis, and a structured field's an
index. `Term` is the protocol every such term implements: it declares the
finite axes, the data inputs and the latent sites the program reads, and the
laws that only its compilation knows. The functions of this module turn those
declarations into program declarations and bindings without knowing which term
made them.

A term over an output axis is the scalar term under a binder over the axis,
with each coefficient read at the bound coordinate. `over_output` and
`at_coordinate` write that rule once, so a term states its mathematics for one
coordinate. The contributions of such a term are bound over the axis first and
the rows second.

An emitted program lists its objects, inputs and prior targets in the order of
the predictors and their terms.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Protocol

import didactic.api as dx
import numpy as np
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
    SampleStep,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _add, _factor, _index, _mul, _var
from quivers.formulas._design import SymbolicColumn

if TYPE_CHECKING:
    from quivers.formulas._row_source import PredictorRows
    from quivers.formulas._group_law import GroupBlockLaw
    from quivers.formulas.formula import FixedColumn

__all__: list[str] = []


class OutputAxis(ValueModel):
    """Coordinates a predictor or term has one coefficient vector for.

    Parameters
    ----------
    name : str
        Finite axis of the coordinates.
    size : int
        Number of coordinates.
    labels : tuple[str, ...]
        Retained label of each coordinate, or empty when coordinates are named
        by position.
    """

    name: str
    size: int
    labels: tuple[str, ...] = ()


class AxisDecl(ValueModel):
    """One finite axis a term uses.

    Parameters
    ----------
    name : str
        Object name.
    size : int
        Number of elements.
    """

    name: str
    size: int


class InputDecl(ValueModel):
    """One data input a term reads.

    Parameters
    ----------
    name : str
        Input name.
    element : {"Real", "Int", "Bool"}
        Element carrier.
    axes : tuple[str, ...]
        Finite axes of the input, outermost first; empty for a scalar.
    declared : bool
        Whether the program declares the input with its type, or the type is
        inferred from its uses.
    values : torch.Tensor or None
        Values on the rows of the term; None when another part of the model
        binds them.
    """

    name: str
    element: Literal["Real", "Int", "Bool"]
    axes: tuple[str, ...]
    declared: bool
    values: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class SiteDecl(ValueModel):
    """One latent site of a term.

    Parameters
    ----------
    name : str
        Site name.
    class_ : str
        brms prior class.
    default : str
        Default prior as QVR source; empty when the term fixes the law.
    coef : str or None
        Authored coefficient label.
    group : str or None
        Authored grouping factor.
    axes : tuple[str, ...]
        Finite axes the site is drawn over.
    event_axis : str or None
        Axis of the event of one draw, for a simplex or a Cholesky factor.
    lower, upper : float or None
        Bounds of the support.
    overridable : bool
        Whether prior rules can select the site.
    fresh : bool
        Whether the site is drawn for fresh rows rather than replayed.
    divisor : float or None
        Root mean square of the design column the coefficient multiplies, which
        divides the scale of a default prior.
    multiplier : float or None
        Factor multiplying the scale of a default prior.
    component_labels : tuple[str, ...]
        Labels of the output coordinates indexing the site.
    partition_labels : tuple[str, ...]
        Labels of the partition of the levels indexing the site.
    """

    name: str
    class_: str
    default: str = ""
    coef: str | None = None
    group: str | None = None
    axes: tuple[str, ...] = ()
    event_axis: str | None = None
    lower: float | None = None
    upper: float | None = None
    overridable: bool = True
    fresh: bool = False
    divisor: float | None = None
    multiplier: float | None = None
    component_labels: tuple[str, ...] = ()
    partition_labels: tuple[str, ...] = ()

    @property
    def index(self) -> str | None:
        """Return the one axis the draw is plated over or takes its event from.

        Returns
        -------
        str or None
            Axis name, or None for a scalar site.

        Raises
        ------
        ValueError
            If the site declares several axes.
        """
        axes = (*self.axes, *((self.event_axis,) if self.event_axis else ()))
        if len(axes) > 1:
            raise ValueError(f"site {self.name!r} is drawn over several axes {axes}")
        return axes[0] if axes else None


class Contribution(ValueModel):
    """One bound summand of a predictor.

    Parameters
    ----------
    name : str
        Binding of the value.
    rows : bool
        Whether the value varies over the rows.
    coordinates : bool
        Whether the value varies over the output axis, which then precedes
        the rows.
    """

    name: str
    rows: bool
    coordinates: bool

    def cell(self, row: LetExprNode, coordinate: LetExprNode) -> LetExprNode:
        """Read the value at one row and output coordinate.

        Parameters
        ----------
        row, coordinate : LetExprNode
            Row and output coordinate.

        Returns
        -------
        LetExprNode
            The binding indexed by the axes it varies over.
        """
        indices = (
            *((coordinate,) if self.coordinates else ()),
            *((row,) if self.rows else ()),
        )
        return _index(self.name, *indices) if indices else _var(self.name)


COORDINATE = "output_coordinate"
"""Binder of the output axis inside the steps of a term."""


def coordinate(output: OutputAxis | None) -> LetExprNode | None:
    """Return the bound output coordinate of a term.

    Parameters
    ----------
    output : OutputAxis or None
        Axis of the term.

    Returns
    -------
    LetExprNode or None
        The binder of the axis, or None for a scalar term.
    """
    return _var(COORDINATE) if output is not None else None


def at_coordinate(
    name: str, output: OutputAxis | None, *indices: LetExprNode
) -> LetExprNode:
    """Read a binding of a term at the bound output coordinate.

    Parameters
    ----------
    name : str
        Binding with one value per coordinate of the axis, outermost.
    output : OutputAxis or None
        Axis of the term.
    *indices : LetExprNode
        Further indices.

    Returns
    -------
    LetExprNode
        The binding at the coordinate and the indices.
    """
    bound = coordinate(output)
    every = (*((bound,) if bound is not None else ()), *indices)
    return _index(name, *every) if every else _var(name)


def over_output(
    output: OutputAxis | None,
    body: LetExprNode,
    *axes: tuple[str, str],
) -> LetExprNode:
    """Bind the output coordinate of an expression.

    Parameters
    ----------
    output : OutputAxis or None
        Axis of the term.
    body : LetExprNode
        Expression at one coordinate.
    *axes : tuple[str, str]
        Further binders and their finite axes, inside the output axis.

    Returns
    -------
    LetExprNode
        The expression over the axes, or the body itself without any.
    """
    binders = (*(((COORDINATE, output.name),) if output is not None else ()), *axes)
    return _factor(binders, body=body) if binders else body


def flat_index(output: OutputAxis | None, position: LetExprNode) -> LetExprNode:
    """Index a flat site holding each element's coordinates consecutively.

    Parameters
    ----------
    output : OutputAxis or None
        Axis of the term.
    position : LetExprNode
        Element of the term's own axis.

    Returns
    -------
    LetExprNode
        ``position * size + coordinate``, or the position of a scalar term.
    """
    bound = coordinate(output)
    if output is None or bound is None:
        return position
    size = LetExprLiteral(value=output.size, integral=True)
    return _add(_mul(position, size), bound)


class Emission(Protocol):
    """What a term asks of the program being emitted."""

    def draw(self, site: SiteDecl) -> SampleStep:
        """Draw a site from its configured or default prior.

        Parameters
        ----------
        site : SiteDecl
            Site with a default prior.

        Returns
        -------
        SampleStep
            The draw over the site's axis.
        """
        ...

    def gathered(
        self, key: tuple[str, str, str], law: GroupBlockLaw | None
    ) -> tuple[tuple[ProgramStep, ...], LetExprNode]:
        """Read a group-level coefficient at the rows' levels.

        Parameters
        ----------
        key : tuple[str, str, str]
            Predictor namespace, grouping factor, and coefficient label.
        law : GroupBlockLaw or None
            Law of the coefficient's block; None when another program emits it.

        Returns
        -------
        tuple[tuple[ProgramStep, ...], LetExprNode]
            The steps of the block, empty once it is emitted, and the gathered
            coefficient.
        """
        ...

    def correlation(
        self, site: str, axis: AxisDecl, result: str
    ) -> tuple[ProgramStep, ...]:
        """Draw a correlation factor over a finite axis.

        Parameters
        ----------
        site : str
            Factor site.
        axis : AxisDecl
            Axis of the factor.
        result : str
            Binding of the factor, constant over an axis of one element.

        Returns
        -------
        tuple[ProgramStep, ...]
            The draw and the binding.
        """
        ...


class Term(Protocol):
    """One additive term of a predictor."""

    @property
    def label(self) -> str:
        """Return the authored term.

        Returns
        -------
        str
            Label of the term in the formula.
        """
        ...

    @property
    def output(self) -> OutputAxis | None:
        """Return the axis of the term's coefficient vectors.

        Returns
        -------
        OutputAxis or None
            The axis, or None for a scalar term.
        """
        ...

    def at(self, rows: PredictorRows) -> Term:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        Term
            The same coefficients over the design of the rows. A site drawn afresh
            for them says so.
        """
        ...

    def contributions(self) -> tuple[Contribution, ...]:
        """List the bound summands of the term.

        Returns
        -------
        tuple[Contribution, ...]
            Values whose sum is the contribution of the term.
        """
        ...

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            Axes of the term's own sites and inputs.
        """
        ...

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs the term reads.

        Returns
        -------
        tuple[InputDecl, ...]
            Inputs with their values on the term's rows.
        """
        ...

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            Sites the term draws, hyperparameters included.
        """
        ...

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            Native laws the term's sites are drawn from, or None.
        """
        ...

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Draw the term's sites and bind its contributions.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            Draws and bindings in dependency order.
        """
        ...


def tensor_type(element: str, *axes: str) -> QiecTypeApplication | QiecTypeName:
    """Spell the type of a tensor over named finite axes.

    Parameters
    ----------
    element : str
        Element carrier.
    *axes : str
        Axis names, outermost first.

    Returns
    -------
    QiecTypeApplication or QiecTypeName
        ``Tensor[element]([axes])``, or the element itself without axes.
    """
    if not axes:
        return QiecTypeName(name=element)
    return QiecTypeApplication(
        constructor="Tensor",
        static_arguments=(QiecTypeName(name=element),),
        indices=(
            QiecShapeIndex(dimensions=tuple(QiecIndexName(name=axis) for axis in axes)),
        ),
    )


def read_only(values: np.ndarray) -> np.ndarray:
    """Copy an array in double precision and clear its writeable flag.

    Parameters
    ----------
    values : np.ndarray
        Numeric values.

    Returns
    -------
    np.ndarray
        Read-only copy.
    """
    copy = np.array(values, dtype=np.float64, order="C")
    copy.setflags(write=False)
    return copy


def column_rms(data: np.ndarray) -> float | None:
    """Return the root mean square a default coefficient scale is divided by.

    Parameters
    ----------
    data : np.ndarray
        Design column.

    Returns
    -------
    float or None
        The root mean square, or None unless it is positive and finite.
    """
    value = float(np.sqrt(np.mean(np.square(np.asarray(data, dtype=np.float64)))))
    return value if np.isfinite(value) and value > 0.0 else None


def column_inputs(
    columns: Iterable[tuple[str, FixedColumn | SymbolicColumn]],
) -> tuple[InputDecl, ...]:
    """Declare the row vectors of design columns.

    Parameters
    ----------
    columns : Iterable[tuple[str, FixedColumn or SymbolicColumn]]
        Input name and column.

    Returns
    -------
    tuple[InputDecl, ...]
        One inferred row vector per evaluated column, and one declared row
        vector, bound elsewhere, per symbolic column. An intercept has none.
    """
    return tuple(
        InputDecl(
            name=name,
            element="Real",
            axes=("Resp",),
            declared=True,
        )
        if isinstance(column, SymbolicColumn)
        else InputDecl(
            name=name,
            element="Real",
            axes=("Resp",),
            declared=False,
            values=torch.as_tensor(column.data.copy(), dtype=torch.float32),
        )
        for name, column in columns
        if not column.is_intercept
    )


def design_declarations(name: str, matrix: np.ndarray) -> tuple[AxisDecl, InputDecl]:
    """Declare the coordinate axis and the row design of a coupled block.

    Parameters
    ----------
    name : str
        Prefixed design identifier.
    matrix : np.ndarray
        Rows by coordinates.

    Returns
    -------
    tuple[AxisDecl, InputDecl]
        The coordinate axis, and the matrix in double precision.
    """
    axis = AxisDecl(name=name + "_Coordinate", size=matrix.shape[1])
    return axis, InputDecl(
        name=name,
        element="Real",
        axes=("Resp", axis.name),
        declared=True,
        values=torch.as_tensor(matrix.copy(), dtype=torch.float64),
    )


def declare_axes(terms: Iterable[Term]) -> tuple[AxisDecl, ...]:
    """List the axes a program declares for its terms.

    Parameters
    ----------
    terms : Iterable[Term]
        Terms of every predictor, in predictor order.

    Returns
    -------
    tuple[AxisDecl, ...]
        Axes in term order.
    """
    return tuple(axis for term in terms for axis in term.axes())


def declare_inputs(inputs: Iterable[InputDecl]) -> tuple[QiecValueParameter, ...]:
    """Declare the typed inputs of a program.

    Parameters
    ----------
    inputs : Iterable[InputDecl]
        Inputs of every term, in predictor order.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        One parameter per declared input, in order of first appearance.

    Raises
    ------
    ValueError
        If two terms declare one input with different types.
    """
    parameters: dict[str, QiecValueParameter] = {}
    for item in inputs:
        if not item.declared:
            continue
        parameter = QiecValueParameter(
            name=item.name, type_expr=tensor_type(item.element, *item.axes)
        )
        if parameters.setdefault(item.name, parameter) != parameter:
            raise ValueError(f"terms declare input {item.name!r} with two types")
    return tuple(parameters.values())


def bind_inputs(inputs: Iterable[InputDecl]) -> dict[str, torch.Tensor]:
    """Bind the inputs whose values the terms hold.

    Parameters
    ----------
    inputs : Iterable[InputDecl]
        Inputs of every term.

    Returns
    -------
    dict[str, torch.Tensor]
        Values by input name.

    Raises
    ------
    ValueError
        If two terms bind one input to different values.
    """
    bound: dict[str, torch.Tensor] = {}
    for item in inputs:
        if item.values is None:
            continue
        values = bound.setdefault(item.name, item.values)
        if values.dtype != item.values.dtype or not torch.allclose(
            values, item.values, rtol=0.0, atol=0.0, equal_nan=True
        ):
            raise ValueError(f"terms bind input {item.name!r} to two values")
    return bound


def catalogue_sites(terms: Iterable[Term]) -> tuple[SiteDecl, ...]:
    """List the sites prior rules can select.

    Parameters
    ----------
    terms : Iterable[Term]
        Terms of one predictor.

    Returns
    -------
    tuple[SiteDecl, ...]
        Overridable sites in term order.
    """
    return tuple(site for term in terms for site in term.sites() if site.overridable)


def join_catalogs(
    base: FamilyCatalog | None, terms: Iterable[Term]
) -> FamilyCatalog | None:
    """Join a response catalog with the laws of every term.

    Parameters
    ----------
    base : FamilyCatalog or None
        Catalog of the response family.
    terms : Iterable[Term]
        Terms of every predictor.

    Returns
    -------
    FamilyCatalog or None
        The joined catalog, or ``base`` when no term has laws of its own.
    """
    catalog = base
    for term in terms:
        laws = term.catalog()
        if laws is not None:
            catalog = laws if catalog is None else catalog.with_catalog(laws)
    return catalog
