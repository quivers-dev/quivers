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
from quivers.formulas._term import Term, VariableDecl
from quivers.formulas._variable_catalogue import (
    SmoothNames,
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
        present = {term.term for term in self.formula.design_terms} | {
            column.term for column in self.formula.fixed_columns
        }
        return smooth_names(
            self.formula,
            {term: state for term, state in self.states.items() if term in present},
            infix=self.infix,
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
        if declaration.index:
            value = value[(..., *declaration.index)]
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
    for term in terms:
        for declaration, values in named_variables(term, scope, draws):
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
