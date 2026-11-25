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

An emitted program lists its objects, inputs and prior targets by *section*,
then by predictor, then by term. A declaration names its section.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Protocol

import didactic.api as dx
import numpy as np
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.dsl.ast_nodes import (
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._design import SymbolicColumn

if TYPE_CHECKING:
    from quivers.formulas.formula import FixedColumn

__all__: list[str] = []

type Section = Literal[
    "field", "special", "simplex", "level", "column", "design", "inline"
]
"""Part of a program's declarations: the inputs and sites of structured fields,
of special-effect columns and their simplexes, of grouping levels, of design
columns, and of coupled design blocks. ``inline`` axes are declared beside the
steps that use them."""

AXIS_SECTIONS: tuple[Section, ...] = ("design", "special")
"""Order of the axes declared ahead of the program body."""

INPUT_SECTIONS: tuple[Section, ...] = (
    "field",
    "special",
    "level",
    "column",
    "design",
)
"""Order of the inputs in the typed telescope of a program."""

SITE_SECTIONS: tuple[Section, ...] = (
    "design",
    "column",
    "special",
    "simplex",
    "level",
    "field",
)
"""Order of the prior targets of one predictor."""


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
    section : Section
        Part of the declarations it belongs to.
    """

    name: str
    size: int
    section: Section


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
    section : Section
        Part of the telescope it belongs to.
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
    section: Section
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
    section : Section
        Part of the prior catalogue it belongs to.
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
    divisor : float or None
        Root mean square of the design column the coefficient multiplies, which
        divides the scale of a default prior.
    multiplier : float or None
        Factor multiplying the scale of a default prior.
    component_labels : tuple[str, ...]
        Labels of the output coordinates indexing the site.
    partition_labels : tuple[str, ...]
        Labels of the partition of the levels indexing the site.
    scoped : bool
        Whether the site belongs to the parameter its predictor predicts.
    rank : int
        Position among the predictor's sites of the same section.
    """

    name: str
    class_: str
    section: Section
    default: str = ""
    coef: str | None = None
    group: str | None = None
    axes: tuple[str, ...] = ()
    event_axis: str | None = None
    lower: float | None = None
    upper: float | None = None
    overridable: bool = True
    divisor: float | None = None
    multiplier: float | None = None
    component_labels: tuple[str, ...] = ()
    partition_labels: tuple[str, ...] = ()
    scoped: bool = True
    rank: int = 0

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

    @property
    def rowwise(self) -> bool:
        """Report whether the contribution varies over the rows.

        Returns
        -------
        bool
            False only for a term with no row-varying design, such as an
            intercept.
        """
        ...

    @property
    def deferred(self) -> bool:
        """Report whether the term is drawn after the auxiliary parameters.

        Returns
        -------
        bool
            True for a term whose steps follow the auxiliary parameters.
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
            section="column",
            declared=True,
        )
        if isinstance(column, SymbolicColumn)
        else InputDecl(
            name=name,
            element="Real",
            axes=("Resp",),
            section="column",
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
    axis = AxisDecl(name=name + "_Coordinate", size=matrix.shape[1], section="design")
    return axis, InputDecl(
        name=name,
        element="Real",
        axes=("Resp", axis.name),
        section="design",
        declared=True,
        values=torch.as_tensor(matrix.copy(), dtype=torch.float64),
    )


def declare_axes(terms: Iterable[Term]) -> tuple[AxisDecl, ...]:
    """List the axes a program declares ahead of its body.

    Parameters
    ----------
    terms : Iterable[Term]
        Terms of every predictor, in predictor order.

    Returns
    -------
    tuple[AxisDecl, ...]
        Axes by section, then in term order.
    """
    axes = [axis for term in terms for axis in term.axes()]
    return tuple(
        axis for section in AXIS_SECTIONS for axis in axes if axis.section == section
    )


def declare_inputs(inputs: Iterable[InputDecl]) -> tuple[QiecValueParameter, ...]:
    """Declare the typed inputs of a program.

    Parameters
    ----------
    inputs : Iterable[InputDecl]
        Inputs of every term, in predictor order.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        One parameter per declared input, by section, then in order of first
        appearance.
    """
    declared = tuple(item for item in inputs if item.declared)
    parameters: dict[str, QiecValueParameter] = {}
    for section in INPUT_SECTIONS:
        for item in declared:
            if item.section == section:
                parameters.setdefault(
                    item.name,
                    QiecValueParameter(
                        name=item.name, type_expr=tensor_type(item.element, *item.axes)
                    ),
                )
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
    """
    return {item.name: item.values for item in inputs if item.values is not None}


def catalogue_sites(terms: Iterable[Term]) -> tuple[SiteDecl, ...]:
    """List the sites prior rules can select.

    Parameters
    ----------
    terms : Iterable[Term]
        Terms of one predictor.

    Returns
    -------
    tuple[SiteDecl, ...]
        Overridable sites by section, then by their rank within it.
    """
    sites = [site for term in terms for site in term.sites() if site.overridable]
    return tuple(
        sorted(sites, key=lambda site: (SITE_SECTIONS.index(site.section), site.rank))
    )


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
