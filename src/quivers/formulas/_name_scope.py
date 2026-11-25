"""brms's names for the coefficients of one predictor.

A `NameScope` holds what a term needs to spell its coefficients as brms does:
the parameter and response parts of the names, the labels of the design columns,
and the labels of the smooth terms. A term over an output axis whose coordinates
are parameters of their own, such as the location of each response category, has
the names of the scalar term in the scope of each coordinate's parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.formulas._smooth import SmoothState
from quivers.formulas._term import OutputAxis, Term, VariableDecl
from quivers.formulas._variable_catalogue import (
    CoefficientName,
    SmoothNames,
    brms_rename,
    scalar_coordinates,
    smooth_names,
)
from quivers.formulas.formula import Formula

__all__: list[str] = []


class NameScope(ValueModel):
    """The parts of brms's names one predictor fixes.

    Parameters
    ----------
    parameter : str
        Predicted parameter.
    role : {"dpar", "nlpar"}
        Whether it is a parameter of the response law or of a nonlinear
        expression.
    response : str
        Response label of a multivariate model, or empty.
    formula : Formula
        Evaluated predictor.
    labels : dict[str, str]
        brms label of each design column, by column name.
    states : Mapping[str, SmoothState]
        Retained states of the predictor's smooth terms, by term label.
    special : dict[str, str]
        brms label of each special-effects column, by column name.
    intercept : bool
        Whether the intercept is a coefficient of its own, not part of the
        thresholds of an ordinal response.
    """

    parameter: str
    role: Literal["dpar", "nlpar"]
    response: str
    formula: Formula = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    labels: dict[str, str]
    states: Mapping[str, SmoothState] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    special: dict[str, str] = {}
    intercept: bool = True

    @property
    def infix(self) -> str:
        """Return the parameter and response part of a coefficient name.

        Returns
        -------
        str
            Empty for the location of a single response. A distributional
            parameter precedes the response and a nonlinear parameter follows
            it, as in brms.
        """
        response = [self.response] if self.response else []
        if self.parameter == "mu" and self.role == "dpar":
            parts = response
        elif self.role == "dpar":
            parts = [self.parameter, *response]
        else:
            parts = [*response, self.parameter]
        return "".join(f"{part}_" for part in parts)

    @property
    def smooths(self) -> SmoothNames:
        """Return the names of the predictor's smooth terms.

        Returns
        -------
        SmoothNames
            Names of the unpenalized columns, the scales and the coefficients.
        """
        # A location whose categories have formulas of their own has no terms.
        stated = self.formula.fixed_columns or self.formula.design_terms
        return smooth_names(
            self.formula, self.states if stated else {}, infix=self.infix
        )

    def label(self, column: str) -> str:
        """Return the label of one design column.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        str
            brms's label, or the column name when formulae evaluated no label.
        """
        return self.labels.get(column, column)

    def coordinate(self, parameter: str) -> NameScope:
        """Return the scope of one coordinate of a vector-valued parameter.

        Parameters
        ----------
        parameter : str
            Parameter of the coordinate, such as ``muB``.

        Returns
        -------
        NameScope
            The same predictor under the coordinate's own parameter.
        """
        return self.with_(parameter=parameter, role="dpar")


def named_variables(
    term: Term, scope: NameScope, draws: Mapping[str, torch.Tensor]
) -> tuple[tuple[VariableDecl, dict[str, torch.Tensor]], ...]:
    """Read the named variables of one term from posterior draws.

    Parameters
    ----------
    term : Term
        Planned term.
    scope : NameScope
        Names of its predictor.
    draws : Mapping[str, torch.Tensor]
        Draws of the term's sites under their local names.

    Returns
    -------
    tuple[tuple[VariableDecl, dict[str, torch.Tensor]], ...]
        Each declaration whose site was drawn, with its scalar variables.
    """
    result = []
    for declaration in term.variables(scope):
        if declaration.site not in draws:
            continue
        if not declaration.name:
            result.append((declaration, {}))
            continue
        value = (
            term.coefficients(draws) if declaration.derived else draws[declaration.site]
        )
        if declaration.flat is not None:
            value = value.unflatten(-1, (-1, declaration.flat))
        if declaration.index:
            event = (slice(None),) * declaration.event_rank
            value = value[(..., *declaration.index, *event)]
        result.append((declaration, scalar_coordinates(declaration.name, value)))
    return tuple(result)


class TermVariables(ValueModel):
    """brms's variables of the terms of one predictor.

    Parameters
    ----------
    population : Mapping[str, torch.Tensor]
        Intercepts and coefficients.
    special : Mapping[str, torch.Tensor]
        Hyperparameters and latent coefficients.
    intercepts : frozenset[str]
        Names of the intercepts.
    sites : dict[str, str]
        Name of each scalar coefficient without its class, by site.
    read : frozenset[str]
        Sites the variables report.
    """

    population: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    special: Mapping[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    intercepts: frozenset[str]
    sites: dict[str, str]
    read: frozenset[str]


def predictor_variables(
    terms: Iterable[Term], scope: NameScope, draws: Mapping[str, torch.Tensor]
) -> TermVariables:
    """Name the draws of one predictor's terms.

    Parameters
    ----------
    terms : Iterable[Term]
        Terms of the predictor.
    scope : NameScope
        Names of the predictor.
    draws : Mapping[str, torch.Tensor]
        Draws under the predictor's local site names.

    Returns
    -------
    TermVariables
        Every variable the terms give a name.
    """
    population: dict[str, torch.Tensor] = {}
    special: dict[str, torch.Tensor] = {}
    intercepts: set[str] = set()
    sites: dict[str, str] = {}
    read: set[str] = set()
    named = [
        (
            declaration.index[-1] if term.output and term.output.parameters else 0,
            declaration,
            values,
        )
        for term in terms
        for declaration, values in named_variables(term, scope, draws)
    ]
    # brms lists the variables of a vector of parameters parameter by parameter.
    for _, declaration, values in sorted(named, key=lambda item: item[0]):
        read.add(declaration.site)
        if declaration.kind == "special":
            special |= values
            continue
        population |= values
        if declaration.kind == "intercept":
            intercepts |= values.keys()
        if not declaration.index and declaration.name:
            sites[declaration.site] = declaration.name.partition("_")[2]
    return TermVariables(
        population=population,
        special=special,
        intercepts=frozenset(intercepts),
        sites=sites,
        read=frozenset(read),
    )


class LevelCoefficient(ValueModel):
    """One scalar group-level coefficient and its brms name.

    Parameters
    ----------
    group : str
        Grouping factor.
    coefficient : CoefficientName
        Its name; a coefficient indexed by position ends in its index.
    values : torch.Tensor
        Draws with chain, draw, and level axes.
    """

    group: str
    coefficient: CoefficientName
    values: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    @property
    def name(self) -> str:
        """Return the label brms's ``ranef`` gives the coefficient.

        Returns
        -------
        str
            The parameter, the response, and the coefficient.
        """
        return self.coefficient.name

    def variable(self, level: str) -> str:
        """Return brms's variable of one level.

        Parameters
        ----------
        level : str
            Level of the grouping factor.

        Returns
        -------
        str
            ``r_<group>__<parameter>[<level>,<coefficient>]``.
        """
        infix = self.coefficient.infix.rstrip("_")
        owner = f"__{infix}" if infix else ""
        return f"r_{self.group}{owner}[{level},{self.coefficient.label}]"


def coefficient_names(
    scope: NameScope, output: OutputAxis | None, coefficient: str, width: int | None
) -> tuple[CoefficientName, ...]:
    """Name the components of one group-level coefficient.

    Parameters
    ----------
    scope : NameScope
        Names of the coefficient's predictor.
    output : OutputAxis or None
        Axis of the predictor.
    coefficient : str
        Design label of the coefficient.
    width : int or None
        Number of components; None for a scalar coefficient.

    Returns
    -------
    tuple[CoefficientName, ...]
        One name per component. The
        components of a vector of parameters have the labels of the scalar
        coefficient in each parameter, and others their position.
    """
    label = scope.special.get(coefficient)
    if label is None:
        label = (
            scope.label(coefficient)
            if coefficient in scope.labels
            else brms_rename(coefficient)
        )
    if width is None:
        return (CoefficientName(infix=scope.infix, label=label),)
    if output is not None and len(output.parameters) == width:
        return tuple(
            CoefficientName(
                infix=scope.coordinate(name).infix, label=label, parameter=index
            )
            for index, name in enumerate(output.parameters)
        )
    return tuple(
        CoefficientName(infix=scope.infix, label=f"{label}[{index + 1}]")
        for index in range(width)
    )


def level_coefficients(
    varying: Mapping[str, Mapping[str, torch.Tensor]],
    scope: NameScope,
    output: OutputAxis | None,
) -> tuple[LevelCoefficient, ...]:
    """Name the group-level coefficients of one predictor.

    Parameters
    ----------
    varying : Mapping[str, Mapping[str, torch.Tensor]]
        Draws by grouping factor and coefficient label, with chain, draw, and
        level axes, and a coordinate axis for coefficient vectors.
    scope : NameScope
        Names of the predictor.
    output : OutputAxis or None
        Axis of the predictor.

    Returns
    -------
    tuple[LevelCoefficient, ...]
        One scalar coefficient per label and coordinate.
    """
    result: list[LevelCoefficient] = []
    for group, coefficients in varying.items():
        for coefficient, values in coefficients.items():
            scalar = values.ndim == 3
            names = coefficient_names(
                scope, output, coefficient, None if scalar else values.shape[-1]
            )
            parts = (values,) if scalar else values.unbind(-1)
            result.extend(
                LevelCoefficient(group=group, coefficient=name, values=part)
                for name, part in zip(names, parts, strict=True)
            )
    # brms lists the coefficients of a factor parameter by parameter.
    groups = list(varying)
    return tuple(
        sorted(
            result,
            key=lambda item: (groups.index(item.group), item.coefficient.parameter),
        )
    )
