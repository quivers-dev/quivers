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

from collections.abc import Iterable
from typing import TYPE_CHECKING

from formulae.matrices import DesignMatrices
import numpy as np
import pandas as pd
import torch

from quivers.dsl.ast_nodes import (
    DrawArgList,
    DrawArgNamed,
    DrawArgScalar,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    ProgramStep,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
    SampleStep,
    TypeName,
)
from quivers.formulas._ast import _call, _let, _mul, _var
from quivers.formulas._monotonic import (
    MonotonicTerm,
    evaluate_monotonic_term,
)
from quivers.formulas.formula import FixedColumn, Formula, _qvr_name

if TYPE_CHECKING:
    from quivers.formulas.compile import FormulaToQVRModule

__all__: list[str] = []


class _Simplex:
    """Names and extent of one simplex site, shared by every factor with its label."""

    def __init__(self, site: str, axis: str, width: int, label: str) -> None:
        """Retain one simplex site.

        Parameters
        ----------
        site, axis : str
            Latent binder and its finite axis.
        width : int
            Simplex dimension.
        label : str
            brms ``simo`` coefficient label.
        """
        self.site = site
        self.axis = axis
        self.width = width
        self.label = label


def simplex_sites(formula: Formula, prefix: str) -> dict[str, _Simplex]:
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
    dict[str, _Simplex]
        Simplex records keyed by brms label, in first-appearance order.

    Raises
    ------
    ValueError
        If factors sharing an ``id`` have different numbers of categories, or
        two labels normalize to one site.
    """
    sites: dict[str, _Simplex] = {}
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
            sites[factor.simplex] = _Simplex(
                site,
                f"FormulaMonotonic_{prefix}{label}",
                factor.width,
                factor.simplex,
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


def monotonic_axes(
    predictors: Iterable[tuple[str, str, Formula]],
) -> tuple[tuple[str, int], ...]:
    """List the finite simplex axes of every predictor.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Axis names and simplex dimensions.
    """
    return tuple(
        (simplex.axis, simplex.width)
        for _, prefix, formula in predictors
        for simplex in simplex_sites(formula, prefix).values()
    )


def _simplex_prior(lens: FormulaToQVRModule, simplex: _Simplex) -> SampleStep:
    """Emit the Dirichlet prior of one simplex, expanding a scalar concentration.

    brms's ``dirichlet(1)`` recycles a scalar concentration to the simplex
    dimension; a vector concentration must have that length.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the default and overriding priors.
    simplex : _Simplex
        Site and extent.

    Returns
    -------
    SampleStep
        The Dirichlet draw over the simplex axis.

    Raises
    ------
    ValueError
        If the prior is not a Dirichlet, or its concentration has another length.
    """
    default = "Dirichlet([" + ", ".join(["1.0"] * simplex.width) + "])"
    step = lens._prior_draw(simplex.site, default, index=TypeName(name=simplex.axis))
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


def append_monotonic_terms(
    lens: FormulaToQVRModule,
    formula: Formula,
    steps: list[ProgramStep],
    *,
    prefix: str = "",
) -> list[LetExprVar]:
    """Emit the simplexes, coefficients, and row contributions of ``mo()`` terms.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler carrying the prior configuration.
    formula : Formula
        One predictor's evaluated terms.
    steps : list[ProgramStep]
        Program steps receiving the draws and bindings.
    prefix : str
        Predictor namespace.

    Returns
    -------
    list[LetExprVar]
        One row contribution per special-effects column.

    Raises
    ------
    ValueError
        If a generated name collides with a design input of the predictor, or
        a simplex prior is invalid.
    """
    if not formula.monotonic_terms:
        return []
    taken = {prefix + column.qvr_name for column in formula.fixed_columns} | {
        f"{prefix}random_{term.column.qvr_name}"
        for term in formula.random_terms
        if term.column is not None
    }
    sites = simplex_sites(formula, prefix)
    for simplex in sites.values():
        steps.append(_simplex_prior(lens, simplex))
    contributions = []
    for term in formula.monotonic_terms:
        value = value_binding(prefix, term)
        if value in taken or coefficient_site(prefix, term) in taken:
            raise ValueError(
                f"mo() term {term.name!r} collides with the design input {value!r}"
            )
        factors: list[LetExprNode] = []
        for position, factor in enumerate(term.factors):
            factors.append(
                _mul(
                    LetExprLiteral(value=float(factor.width)),
                    _call(
                        "matmul",
                        _var(_design_input(prefix, term, position)),
                        _var(sites[factor.simplex].site),
                    ),
                )
            )
        if term.column >= 0:
            factors.append(_var(_covariate_input(prefix, term)))
        product = factors[0]
        for factor_value in factors[1:]:
            product = _mul(product, factor_value)
        steps.append(_let(value, product))
        coefficient = coefficient_site(prefix, term)
        steps.append(
            lens._prior_draw(
                coefficient,
                lens._fixed_prior,
                column=FixedColumn(
                    term=term.term,
                    name=term.name,
                    qvr_name=_qvr_name(term.name),
                    data=term.linear_values,
                ),
            )
        )
        contribution = coefficient + "_per_row"
        steps.append(_let(contribution, _mul(_var(coefficient), _var(value))))
        contributions.append(_var(contribution))
    return contributions


def _real_tensor(*axes: str) -> QiecTypeApplication:
    """Declare a Real tensor over named finite axes.

    Parameters
    ----------
    *axes : str
        Axis names in order.

    Returns
    -------
    QiecTypeApplication
        The checked tensor type.
    """
    return QiecTypeApplication(
        constructor="Tensor",
        static_arguments=(QiecTypeName(name="Real"),),
        indices=(
            QiecShapeIndex(dimensions=tuple(QiecIndexName(name=axis) for axis in axes)),
        ),
    )


def monotonic_runtime_parameters(
    predictors: Iterable[tuple[str, str, Formula]],
) -> tuple[QiecValueParameter, ...]:
    """Declare the indicator matrices and covariates of every ``mo()`` column.

    Parameters
    ----------
    predictors : Iterable[tuple[str, str, Formula]]
        Parameter labels, prefixes, and evaluated formulas.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        Real matrix inputs over the row axis and each simplex axis, and Real
        row vectors for the covariates.
    """
    parameters = []
    for _, prefix, formula in predictors:
        sites = simplex_sites(formula, prefix)
        for term in formula.monotonic_terms:
            for position, factor in enumerate(term.factors):
                parameters.append(
                    QiecValueParameter(
                        name=_design_input(prefix, term, position),
                        type_expr=_real_tensor("Resp", sites[factor.simplex].axis),
                    )
                )
            if term.column >= 0:
                parameters.append(
                    QiecValueParameter(
                        name=_covariate_input(prefix, term),
                        type_expr=_real_tensor("Resp"),
                    )
                )
    return tuple(parameters)


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


def monotonic_observations(formula: Formula, prefix: str) -> dict[str, torch.Tensor]:
    """Bind the training inputs of one predictor's ``mo()`` terms.

    Parameters
    ----------
    formula : Formula
        One predictor's evaluated terms.
    prefix : str
        Predictor namespace.

    Returns
    -------
    dict[str, torch.Tensor]
        Indicator matrices and covariates.
    """
    values: dict[str, torch.Tensor] = {}
    for term in formula.monotonic_terms:
        values.update(
            _bindings(
                prefix,
                term,
                tuple(factor.design for factor in term.factors),
                term.covariate if term.column >= 0 else None,
            )
        )
    return values


def monotonic_prediction_bindings(
    formula: Formula, design: DesignMatrices, frame: pd.DataFrame, prefix: str
) -> dict[str, torch.Tensor]:
    """Evaluate one predictor's ``mo()`` inputs on fresh rows.

    Parameters
    ----------
    formula : Formula
        Fitted predictor.
    design : DesignMatrices
        Its retained formulae evaluator.
    frame : pandas.DataFrame
        Fresh rows.
    prefix : str
        Predictor namespace.

    Returns
    -------
    dict[str, torch.Tensor]
        Indicator matrices and covariates of the fresh rows.

    Raises
    ------
    ValueError
        If a fresh value lies outside the training categories, as brms refuses.
    """
    values: dict[str, torch.Tensor] = {}
    if not formula.monotonic_terms:
        return values
    assert design.common is not None
    for term in formula.monotonic_terms:
        components = design.common.terms[term.term].components
        designs, covariate = evaluate_monotonic_term(term, components, frame)
        values.update(_bindings(prefix, term, designs, covariate))
    return values


def monotonic_prior_sites(
    formula: Formula, prefix: str
) -> tuple[tuple[str, str, str], ...]:
    """List the overrideable sites of one predictor's ``mo()`` terms.

    Parameters
    ----------
    formula : Formula
        One predictor's evaluated terms.
    prefix : str
        Predictor namespace.

    Returns
    -------
    tuple[tuple[str, str, str], ...]
        Site, brms class (``b`` or ``simo``), and coefficient label.
    """
    sites: list[tuple[str, str, str]] = [
        (coefficient_site(prefix, term), "b", term.name)
        for term in formula.monotonic_terms
    ]
    sites.extend(
        (simplex.site, "simo", simplex.label)
        for simplex in simplex_sites(formula, prefix).values()
    )
    return tuple(sites)
