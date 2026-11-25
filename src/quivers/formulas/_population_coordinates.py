"""brms's centered intercept and QR coordinates for population-level effects.

brms samples a population-level predictor in transformed coordinates and
reports the coefficients on the original scale:

* centering (``bf(center = TRUE)``, the brms default for location and
  distributional predictors with an intercept) samples the intercept of the
  column-centered design, ``Intercept_c``, and reports
  ``b_Intercept = Intercept_c - sum_j mean(x_j) b_j``. Priors of class
  ``Intercept`` act on ``Intercept_c``;
* ``decomp = "QR"`` writes the (centered) design as ``X = Q R`` with ``R``
  upper triangular with positive diagonal, scales ``XQ = Q sqrt(N - 1)`` and
  ``XR = R / sqrt(N - 1)``, samples ``bQ`` and reports ``b = XR^{-1} bQ``.
  Priors of class ``b`` act on ``bQ`` coordinate by coordinate.

The emitted program samples these coordinates and binds the reported
coefficients deterministically, so the linear predictor, prediction on new
rows, and every reader of ``draws()`` use the original-scale coefficients.
Both maps are linear with constant Jacobian (one for centering,
``|det XR^{-1}|`` for QR), so the model is brms's exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import didactic.api as dx
import numpy as np

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    LetStep,
    ProgramStep,
    SampleStep,
)
from quivers.formulas._ast import _add, _let, _mul, _sub, _var
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._design import SymbolicColumn
from quivers.formulas._term import (
    AxisDecl,
    InputDecl,
    OutputAxis,
    SiteDecl,
    column_inputs,
    column_rms,
)
from quivers.formulas.formula import FixedColumn, Formula

CENTERED_SUFFIX = "_centered"
"""Suffix of the sampled centered-intercept site."""

QR_SUFFIX = "_qr"
"""Suffix of each sampled QR coordinate site."""


class PopulationCoordinates(ValueModel):
    """The sampled coordinates of one predictor's population-level effects.

    Parameters
    ----------
    prefix : str
        Predictor namespace (``""`` for the location, ``dpar_<name>_``).
    centered : bool
        Whether the intercept is sampled on the centered design.
    columns : tuple[str, ...]
        QVR names of the transformed design columns, in design order.
    means : tuple[float, ...]
        Their training means (zeros when not centered).
    r_inverse : tuple[tuple[float, ...], ...] or None
        ``XR^{-1}`` for QR coordinates, or None.
    q_columns : tuple[tuple[float, ...], ...] or None
        Columns of ``XQ``, used to scale default coordinate priors.
    """

    prefix: str
    centered: bool
    columns: tuple[str, ...] = ()
    means: tuple[float, ...] = ()
    r_inverse: tuple[tuple[float, ...], ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    q_columns: tuple[tuple[float, ...], ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _design_columns(formula: Formula) -> list[FixedColumn]:
    """Select the columns brms's population design ``X`` holds.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.

    Returns
    -------
    list[FixedColumn]
        Numeric, non-intercept, non-category-specific columns that are not
        the fixed part of a smooth (brms keeps those in ``Xs``).
    """
    smooth_terms = {term.term for term in formula.design_terms}
    return [
        column
        for column in formula.fixed_columns
        if isinstance(column, FixedColumn)
        and not column.is_intercept
        and not column.category_specific
        and column.term not in smooth_terms
    ]


def thin_qr(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Stan's scaled thin QR factors of a full-rank design.

    Parameters
    ----------
    design : np.ndarray
        ``(N, K)`` design with ``N > K`` and full column rank.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``XQ = Q sqrt(N - 1)`` and ``XR = R / sqrt(N - 1)``, with ``R``'s
        diagonal positive as in Stan's ``qr_thin_Q`` and ``qr_thin_R``, so
        ``XQ @ XR == design``.

    Raises
    ------
    ValueError
        If the design is not of full column rank or has too few rows.
    """
    rows, width = design.shape
    if rows <= width:
        raise ValueError(
            f"decomp='QR' needs more rows than predictors ({rows} rows, {width} columns)"
        )
    q, r = np.linalg.qr(design, mode="reduced")
    signs = np.sign(np.diag(r))
    if np.any(signs == 0) or np.min(np.abs(np.diag(r))) <= 1e-10 * np.max(
        np.abs(np.diag(r))
    ):
        raise ValueError("decomp='QR' requires a design of full column rank")
    q = q * signs
    r = signs[:, None] * r
    scale = np.sqrt(rows - 1.0)
    return q * scale, r / scale


def population_coordinates(
    formula: Formula,
    prefix: str,
    *,
    center: bool,
    decomp: str,
) -> PopulationCoordinates | None:
    """Plan the sampled coordinates of one predictor's population effects.

    Parameters
    ----------
    formula : Formula
        Evaluated training predictor.
    prefix : str
        Predictor namespace.
    center : bool
        Center the design and sample the centered intercept.
    decomp : {"none", "QR"}
        Optional QR coordinates.

    Returns
    -------
    PopulationCoordinates or None
        None when neither transformation applies (no intercept to center and
        no QR), so the ordinary emission is used.

    Raises
    ------
    ValueError
        If QR is requested without non-intercept predictors, with symbolic
        (missing-value) columns, or on a rank-deficient design.
    """
    columns = _design_columns(formula)
    has_intercept = any(column.is_intercept for column in formula.fixed_columns)
    centered = center and has_intercept
    if decomp == "QR":
        if not columns:
            raise ValueError("QR decomposition requires non-intercept predictors")
        if any(isinstance(column, SymbolicColumn) for column in formula.fixed_columns):
            raise ValueError(
                "decomp='QR' needs observed design columns; mi() predictors "
                "have no fixed values"
            )
    if not centered and decomp != "QR":
        return None
    selected = list(columns)
    if decomp == "QR" and not centered and has_intercept:
        # brms keeps an uncentered intercept inside X, so QR includes it.
        selected = [
            column for column in formula.fixed_columns if column.is_intercept
        ] + selected
    data = (
        np.column_stack(
            [
                np.ones(len(formula.response_values))
                if column.is_intercept
                else np.asarray(column.data, dtype=np.float64)
                for column in selected
            ]
        )
        if selected
        else np.zeros((len(formula.response_values), 0))
    )
    means = data.mean(axis=0) if centered else np.zeros(data.shape[1])
    r_inverse = q_columns = None
    if decomp == "QR":
        xq, xr = thin_qr(data - means)
        r_inverse = tuple(tuple(float(v) for v in row) for row in np.linalg.inv(xr))
        q_columns = tuple(tuple(float(v) for v in column) for column in xq.T)
    return PopulationCoordinates(
        prefix=prefix,
        centered=centered,
        columns=tuple(
            "intercept" if column.is_intercept else column.qvr_name
            for column in selected
        ),
        means=tuple(float(value) for value in means),
        r_inverse=r_inverse,
        q_columns=q_columns,
    )


def coefficient_site(prefix: str, column: str) -> str:
    """Return the reported coefficient site of one design column.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    column : str
        QVR column name, or ``intercept``.

    Returns
    -------
    str
        ``<prefix>intercept`` or ``<prefix>beta_<column>``.
    """
    return prefix + ("intercept" if column == "intercept" else f"beta_{column}")


def coordinate_bindings(plan: PopulationCoordinates) -> list[LetStep]:
    """Bind the reported coefficients from the sampled coordinates.

    Parameters
    ----------
    plan : PopulationCoordinates
        The predictor's coordinate plan.

    Returns
    -------
    list[LetStep]
        ``b_j = sum_k XR^{-1}[j, k] bQ_k`` for QR columns, then the
        original-scale intercept ``Intercept_c - sum_j mean_j b_j``.
    """
    steps: list[LetStep] = []
    if plan.r_inverse is not None:
        for j, column in enumerate(plan.columns):
            terms: list[LetExprNode] = [
                _mul(
                    LetExprLiteral(value=plan.r_inverse[j][k]),
                    _var(coefficient_site(plan.prefix, other) + QR_SUFFIX),
                )
                for k, other in enumerate(plan.columns)
                if plan.r_inverse[j][k] != 0.0
            ]
            steps.append(_let(coefficient_site(plan.prefix, column), _add(*terms)))
    if plan.centered:
        shift: list[LetExprNode] = [
            _mul(
                LetExprLiteral(value=mean), _var(coefficient_site(plan.prefix, column))
            )
            for column, mean in zip(plan.columns, plan.means, strict=True)
            if column != "intercept" and mean != 0.0
        ]
        intercept = _var(plan.prefix + "intercept" + CENTERED_SUFFIX)
        steps.append(
            _let(
                plan.prefix + "intercept",
                _sub(intercept, _add(*shift)) if shift else intercept,
            )
        )
    return steps


class PopulationTerm(ValueModel):
    """The population-level columns of one predictor over one output axis.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
    prior : str
        Default prior of a coefficient.
    columns : tuple[FixedColumn or SymbolicColumn, ...]
        Design columns, in authored order.
    ranks : tuple[int, ...]
        Position of each column in the formula.
    coordinates : PopulationCoordinates or None
        Centered and QR coordinates of the block; None samples the
        coefficients themselves.
    output : OutputAxis or None
        Axis of the coefficient vectors.
    """

    prefix: str
    prior: str
    columns: tuple[FixedColumn | SymbolicColumn, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    ranks: tuple[int, ...]
    coordinates: PopulationCoordinates | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    output: OutputAxis | None = None

    @property
    def label(self) -> str:
        """Return the authored terms of the columns.

        Returns
        -------
        str
            Their labels, joined by plus signs.
        """
        return " + ".join(dict.fromkeys(column.term for column in self.columns))

    @property
    def rowwise(self) -> bool:
        """Report whether the contribution varies over the rows.

        Returns
        -------
        bool
            True when the block has a column besides the intercept.
        """
        return any(not column.is_intercept for column in self.columns)

    @property
    def deferred(self) -> bool:
        """Report whether the term is drawn after the auxiliary parameters.

        Returns
        -------
        bool
            False.
        """
        return False

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            Empty: the coefficients use the predictor's own axis.
        """
        return ()

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the design columns.

        Returns
        -------
        tuple[InputDecl, ...]
            One inferred row vector per evaluated column, and one declared
            row vector, bound elsewhere, per symbolic column.
        """
        return column_inputs(
            (self.prefix + column.qvr_name, column) for column in self.columns
        )

    def site(self, column: FixedColumn | SymbolicColumn, rank: int) -> SiteDecl:
        """Declare the sampled coordinate of one column.

        Parameters
        ----------
        column : FixedColumn or SymbolicColumn
            Column of the block.
        rank : int
            Its position in the formula.

        Returns
        -------
        SiteDecl
            The coefficient, the centered intercept, or the QR coordinate.
        """
        plan = self.coordinates
        name = "intercept" if column.is_intercept else column.qvr_name
        site = coefficient_site(self.prefix, name)
        divisor = multiplier = None
        if plan is not None and column.is_intercept and plan.centered:
            site += CENTERED_SUFFIX
        elif plan is not None and plan.q_columns is not None and name in plan.columns:
            # A default prior scales with the XQ column its coordinate multiplies.
            site += QR_SUFFIX
            divisor = column_rms(
                np.asarray(plan.q_columns[plan.columns.index(name)], dtype=np.float64)
            )
        elif isinstance(column, SymbolicColumn):
            multiplier = column.prior_multiplier
        elif not column.is_intercept:
            divisor = column_rms(column.data)
        return SiteDecl(
            name=site,
            class_="Intercept" if column.is_intercept else "b",
            section="column",
            default=self.prior,
            coef=column.name,
            axes=(self.output.name,) if self.output is not None else (),
            divisor=divisor,
            multiplier=multiplier,
            rank=rank,
        )

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the sampled coordinates.

        Returns
        -------
        tuple[SiteDecl, ...]
            One site per column, in authored order.
        """
        return tuple(
            self.site(column, rank)
            for column, rank in zip(self.columns, self.ranks, strict=True)
        )

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            None.
        """
        return None


def sampled_site(site: str, samples: Collection[str]) -> str:
    """Map a reported coefficient site to the site actually sampled.

    Parameters
    ----------
    site : str
        Reported site (``intercept`` or ``beta_<column>`` with a prefix).
    samples : collection of str
        Sampled site names of the emitted program.

    Returns
    -------
    str
        The site itself, its centered intercept, or its QR coordinate.
    """
    if site in samples:
        return site
    for suffix in (CENTERED_SUFFIX, QR_SUFFIX):
        if site + suffix in samples:
            return site + suffix
    return site


def reported_coefficients(formulas: Sequence[tuple[str, str, Formula]]) -> list[str]:
    """List the reported population coefficient sites of every predictor.

    Parameters
    ----------
    formulas : Sequence[tuple[str, str, Formula]]
        Predictor labels, prefixes, and evaluated formulas.

    Returns
    -------
    list[str]
        Intercept and slope sites by prefix.
    """
    return [
        prefix + ("intercept" if column.is_intercept else f"beta_{column.qvr_name}")
        for _, prefix, formula in formulas
        for column in formula.fixed_columns
    ]


__all__: list[str] = []


def emit_population(
    formula: Formula,
    steps: list[ProgramStep],
    prefix: str,
    plan: PopulationCoordinates,
    draw: Callable[[str, FixedColumn | SymbolicColumn], SampleStep],
) -> list[LetExprNode]:
    """Emit population effects in brms's sampled coordinates.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    steps : list
        Program steps receiving draws, bindings, and contributions.
    prefix : str
        Predictor namespace.
    plan : PopulationCoordinates
        Centering and QR plan of this predictor.
    draw : Callable
        The compiler's prior emission for a coefficient site and its column,
        which applies explicit priors and default autoscaling.

    Returns
    -------
    list[LetExprNode]
        Additive contributions on the original design, ``b_Intercept +
        sum_j b_j x_j``; scoring costs O(N K) as for the ordinary emission.
    """
    transformed = set(plan.columns)
    q_columns = dict(zip(plan.columns, plan.q_columns or (), strict=False))
    for column in formula.fixed_columns:
        name = "intercept" if column.is_intercept else column.qvr_name
        site = coefficient_site(prefix, name)
        if column.is_intercept and plan.centered:
            steps.append(draw(site + CENTERED_SUFFIX, column))
        elif plan.r_inverse is not None and name in transformed:
            # Default coordinate priors scale with the XQ column they multiply.
            coordinate = FixedColumn(
                term=column.term,
                name=column.name,
                qvr_name=column.qvr_name,
                data=np.asarray(q_columns[name], dtype=np.float64),
            )
            steps.append(draw(site + QR_SUFFIX, coordinate))
        else:
            steps.append(draw(site, column))
    steps.extend(coordinate_bindings(plan))
    contributions: list[LetExprNode] = []
    for column in formula.fixed_columns:
        site = coefficient_site(
            prefix, "intercept" if column.is_intercept else column.qvr_name
        )
        if column.is_intercept:
            contributions.append(_var(site))
            continue
        contribution = f"{site}_per_row"
        steps.append(
            _let(contribution, _mul(_var(site), _var(prefix + column.qvr_name)))
        )
        contributions.append(_var(contribution))
    return contributions
