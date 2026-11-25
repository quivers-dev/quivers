"""Emit, bind, and catalogue brms ``mo()`` terms through the shared formula compiler.

Each `MonotonicTerm` of a predictor contributes

    bsp * prod_j (D_j * matmul(design_j, simo_j)) * covariate

to that predictor, where ``simo_j : FormulaMonotonic_<label> <- Dirichlet(con)``
is a simplex latent over a finite axis of extent ``D_j`` and ``design_j`` is the
retained row-by-step cumulative indicator matrix. Scoring and prediction cost
O(rows times the sum of the simplex dimensions) per term. A group-level
``(mo(x) | g)`` slope multiplies its group coefficient by the same product,
without ``bsp``, so brms's ``(bsp + r_g) * mo(simo, x)`` is the sum of the two
contributions.
"""

from __future__ import annotations

from collections.abc import Mapping

from functools import reduce
from typing import TYPE_CHECKING

import didactic.api as dx
import numpy as np
import torch

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    DrawArgList,
    DrawArgNamed,
    DrawArgScalar,
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
    SampleStep,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _call, _let, _mul, _var
from quivers.formulas._monotonic import (
    MonotonicTerm,
    evaluate_monotonic_term,
)
from quivers.formulas._row_source import PredictorRows
from quivers.formulas._variable_catalogue import brms_rename
from quivers.formulas._term import (
    AxisDecl,
    VariableDecl,
    SmoothPart,
    coordinate_scopes,
    ColumnName,
    Contribution,
    Emission,
    InputDecl,
    OutputAxis,
    SiteDecl,
    column_rms,
)
from quivers.formulas.formula import Formula, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas._name_scope import NameScope

    pass

__all__: list[str] = []


class Simplex(ValueModel):
    """Names and extent of one simplex site, shared by every factor with its label.

    Parameters
    ----------
    site, axis : str
        Latent binder and its finite axis.
    width : int
        Simplex dimension.
    label : str
        brms ``simo`` coefficient label.
    """

    site: str
    axis: str
    width: int
    label: str


def simplex_sites(formula: Formula, prefix: str) -> dict[str, Simplex]:
    """Resolve the simplex site of every ``mo()`` factor of one predictor.

    Factors with the same brms ``id`` share one site, as brms copies
    ``simo_1`` into every later appearance of the ``id``.

    Parameters
    ----------
    formula : Formula
        One predictor's evaluated terms.
    prefix : str
        Predictor namespace.

    Returns
    -------
    dict[str, Simplex]
        Simplex records keyed by brms label, in first-appearance order.

    Raises
    ------
    ValueError
        If factors sharing an ``id`` have different numbers of categories, or
        two labels normalize to one site.
    """
    sites: dict[str, Simplex] = {}
    names: dict[str, str] = {}
    for term in formula.monotonic_terms:
        for factor in term.factors:
            existing = sites.get(factor.simplex)
            if existing is not None:
                if existing.width != factor.width or not factor.shared:
                    raise ValueError(
                        f"mo() terms sharing id {factor.simplex!r} must have the same "
                        "number of categories"
                        if factor.shared
                        else f"mo() simplex label {factor.simplex!r} is ambiguous"
                    )
                continue
            label = _qvr_name(factor.simplex)
            site = f"{prefix}simo_{label}"
            if names.setdefault(site, factor.simplex) != factor.simplex:
                raise ValueError(
                    f"mo() simplex labels {names[site]!r} and {factor.simplex!r} "
                    "normalize to the same parameter name"
                )
            sites[factor.simplex] = Simplex(
                site=site,
                axis=f"FormulaMonotonic_{prefix}{label}",
                width=factor.width,
                label=factor.simplex,
            )
    return sites


def _design_input(prefix: str, term: MonotonicTerm, position: int) -> str:
    """Name the runtime cumulative-indicator input of one factor.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : MonotonicTerm
        Owning column.
    position : int
        Factor position within the term.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{prefix}{_qvr_name(term.name)}_mo{position + 1}"


def _covariate_input(prefix: str, term: MonotonicTerm) -> str:
    """Name the runtime covariate input of one column.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : MonotonicTerm
        Owning column.

    Returns
    -------
    str
        Runtime input name.
    """
    return f"{prefix}{_qvr_name(term.name)}_covariate"


def coefficient_site(prefix: str, term: MonotonicTerm) -> str:
    """Name the ``bsp`` coefficient site of one column.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : MonotonicTerm
        Owning column.

    Returns
    -------
    str
        Latent binder.
    """
    return f"{prefix}bsp_{_qvr_name(term.name)}"


def value_binding(prefix: str, term: MonotonicTerm) -> str:
    """Name the binding of a column's value before ``bsp``.

    A group-level slope on the same term reads this binding, so it is the
    name the ordinary group emitter derives from the slope label.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : MonotonicTerm
        Owning column.

    Returns
    -------
    str
        Deterministic binding name.
    """
    return prefix + _qvr_name(term.name)


def _recycled_simplex(step: SampleStep, simplex: Simplex) -> SampleStep:
    """Expand a scalar concentration of one simplex's Dirichlet prior.

    brms's ``dirichlet(1)`` recycles a scalar concentration to the simplex
    dimension; a vector concentration must have that length.

    Parameters
    ----------
    step : SampleStep
        Draw of the simplex.
    simplex : Simplex
        Site and extent.

    Returns
    -------
    SampleStep
        The draw with one concentration per coordinate.

    Raises
    ------
    ValueError
        If the prior is not a Dirichlet, or its concentration has another length.
    """
    if step.morphism != "Dirichlet":
        raise ValueError(
            f"the simo prior of {simplex.label!r} must be a Dirichlet distribution, "
            f"got {step.morphism}"
        )
    arguments = []
    for argument in step.args or ():
        if isinstance(argument, DrawArgNamed) and argument.parameter == "concentration":
            value = argument.value
            if isinstance(value, DrawArgScalar):
                value = DrawArgList(items=(value,) * simplex.width)
            if not isinstance(value, DrawArgList) or len(value.items) != simplex.width:
                raise ValueError(
                    f"the Dirichlet concentration for simo {simplex.label!r} must be "
                    f"a scalar or have length {simplex.width}"
                )
            argument = argument.with_(value=value)
        arguments.append(argument)
    return step.with_(args=tuple(arguments))


def _bindings(
    prefix: str,
    term: MonotonicTerm,
    designs: tuple[np.ndarray, ...],
    covariate: np.ndarray | None,
) -> dict[str, torch.Tensor]:
    """Bind one column's indicator matrices and covariate.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : MonotonicTerm
        Owning column.
    designs : tuple[numpy.ndarray, ...]
        Indicator matrix of each factor.
    covariate : numpy.ndarray or None
        Covariate, when the term has one.

    Returns
    -------
    dict[str, torch.Tensor]
        Runtime bindings.
    """
    values = {
        _design_input(prefix, term, position): torch.as_tensor(
            np.array(design, dtype=np.float64)
        )
        for position, design in enumerate(designs)
    }
    if covariate is not None:
        values[_covariate_input(prefix, term)] = torch.as_tensor(
            np.array(covariate, dtype=np.float64)
        )
    return values


class MonotonicRow(ValueModel):
    """One special-effects column: a coefficient times a monotonic row value.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    term : MonotonicTerm
        Evaluated column.
    prior : str
        Default prior of the coefficient.
    scale : float or None
        Root mean square of the training values the coefficient multiplies.
    simplexes : tuple[Simplex, ...]
        Simplex of each factor of the column.
    owned : tuple[Simplex, ...]
        Simplexes drawn with this column.
    output : OutputAxis or None
        None; a column has one scalar coefficient.
    """

    prefix: str
    term: MonotonicTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    prior: str
    scale: float | None
    simplexes: tuple[Simplex, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    owned: tuple[Simplex, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    output: OutputAxis | None = None

    @property
    def label(self) -> str:
        """Return the authored term.

        Returns
        -------
        str
            Label of the column's term.
        """
        return self.term.term

    def at(self, rows: PredictorRows) -> MonotonicRow:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        MonotonicRow
            The same coefficient over the rows' indicator matrices.
        """
        if rows.frame is None:
            return self
        designs, covariate = evaluate_monotonic_term(
            self.term, rows.components(self.term.term), rows.frame
        )
        return self.with_(
            term=self.term.with_(
                factors=tuple(
                    factor.with_(design=design)
                    for factor, design in zip(self.term.factors, designs, strict=True)
                ),
                covariate=self.term.covariate if covariate is None else covariate,
            )
        )

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
            The coefficient, and the simplexes drawn with the column.
        """
        site = coefficient_site(self.prefix, self.term)
        return tuple(
            declaration
            for index, part in coordinate_scopes(self.output, scope)
            for declaration in (
                VariableDecl(
                    name=f"bsp_{part.infix}{brms_rename(self.term.name)}",
                    site=site,
                    kind="population",
                    index=index,
                ),
                *(
                    VariableDecl(
                        name=f"simo_{part.infix}{brms_rename(simplex.label)}",
                        site=simplex.site,
                        kind="special",
                        index=index,
                    )
                    for simplex in self.owned
                ),
            )
        )

    def coefficients(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the coefficients of the term for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        torch.Tensor
            The coefficient.
        """
        return draws[coefficient_site(self.prefix, self.term)]

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

    def contributions(self) -> tuple[Contribution, ...]:
        """List the bound summands of the term.

        Returns
        -------
        tuple[Contribution, ...]
            The row contribution.
        """
        return (
            Contribution(
                name=coefficient_site(self.prefix, self.term) + "_per_row",
                rows=True,
                coordinates=self.output is not None,
            ),
        )

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            The axis of each simplex the column draws.
        """
        return tuple(
            AxisDecl(name=simplex.axis, size=simplex.width) for simplex in self.owned
        )

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs the term reads.

        Returns
        -------
        tuple[InputDecl, ...]
            The cumulative indicator matrix of each factor and the covariate.
        """
        covariate = self.term.covariate if self.term.column >= 0 else None
        values = _bindings(
            self.prefix,
            self.term,
            tuple(factor.design for factor in self.term.factors),
            covariate,
        )
        axes = {
            _design_input(self.prefix, self.term, position): ("Resp", simplex.axis)
            for position, simplex in enumerate(self.simplexes)
        }
        return tuple(
            InputDecl(
                name=name,
                element="Real",
                axes=axes.get(name, ("Resp",)),
                declared=True,
                values=value,
            )
            for name, value in values.items()
        )

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            The simplexes drawn with the column, then its coefficient.
        """
        return (
            *(
                SiteDecl(
                    name=simplex.site,
                    class_="simo",
                    default="Dirichlet([" + ", ".join(["1.0"] * simplex.width) + "])",
                    coef=simplex.label,
                    event_axis=simplex.axis,
                )
                for simplex in self.owned
            ),
            SiteDecl(
                name=coefficient_site(self.prefix, self.term),
                class_="b",
                default=self.prior,
                coef=self.term.name,
                divisor=self.scale,
            ),
        )

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
            The simplexes drawn with the column, its value, its coefficient,
            and its contribution.
        """
        value = value_binding(self.prefix, self.term)
        coefficient = coefficient_site(self.prefix, self.term)
        declared = {site.name: site for site in self.sites()}
        factors: list[LetExprNode] = [
            _mul(
                LetExprLiteral(value=float(simplex.width)),
                _call(
                    "matmul",
                    _var(_design_input(self.prefix, self.term, position)),
                    _var(simplex.site),
                ),
            )
            for position, simplex in enumerate(self.simplexes)
        ]
        if self.term.column >= 0:
            factors.append(_var(_covariate_input(self.prefix, self.term)))
        return (
            *(
                _recycled_simplex(emission.draw(declared[simplex.site]), simplex)
                for simplex in self.owned
            ),
            _let(value, reduce(_mul, factors)),
            emission.draw(declared[coefficient]),
            _let(coefficient + "_per_row", _mul(_var(coefficient), _var(value))),
        )


def monotonic_rows(
    formula: Formula, prefix: str, *, prior: str, output: OutputAxis | None
) -> tuple[MonotonicRow, ...]:
    """Build the special-effects columns of one predictor.

    A simplex is drawn with the first column that reads it, so columns
    sharing an ``id`` read one site.

    Parameters
    ----------
    formula : Formula
        One predictor's evaluated terms.
    prefix : str
        Predictor namespace.
    prior : str
        Default prior of a coefficient.
    output : OutputAxis or None
        Axis of the predictor.

    Returns
    -------
    tuple[MonotonicRow, ...]
        One term per column, in authored order.

    Raises
    ------
    ValueError
        If the predictor has an output axis.
    """
    if not formula.monotonic_terms:
        return ()
    if output is not None:
        raise ValueError(
            "mo() terms are not supported in predictors that carry one "
            "coefficient per response category"
        )
    sites = simplex_sites(formula, prefix)
    rows = []
    drawn: set[str] = set()
    for term in formula.monotonic_terms:
        rows.append(
            MonotonicRow(
                prefix=prefix,
                term=term,
                prior=prior,
                scale=column_rms(term.linear_values),
                simplexes=tuple(sites[factor.simplex] for factor in term.factors),
                owned=tuple(
                    site
                    for name, site in sites.items()
                    if name not in drawn
                    and name in {factor.simplex for factor in term.factors}
                ),
            )
        )
        drawn.update(factor.simplex for factor in term.factors)
    return tuple(rows)
