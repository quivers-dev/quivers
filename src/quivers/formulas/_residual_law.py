"""The residual law of a response: natural residuals correlated across rows.

A family that declares an elliptical residual law (`Family.elliptical`) writes
its response as ``y = mu + sigma * e`` with spherical ``e``. A residual term
(``ar``, ``ma``, ``arma``, ``cosy``, ``unstr``, ``sar``, ``fcor``) correlates
``e`` across the rows of the response, or moves the location by a conditional
ARMA recursion. `ResidualLaw` holds that structure beside the family: the
family stays what the user wrote, and the law supplies the native kernel of the
response, its structural sites, and its form on other rows.

The native kernel is a `ResidualResponse` over the rows of one layout,
registered under ``ResidualNormal_<y>`` or ``ResidualStudentT_<y>`` in a
compilation-local catalog. Its signature is the same for every row set, so a
law on prediction rows changes the catalog and not the program text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Literal

import didactic.api as dx
import narwhals as nw
from narwhals.typing import IntoDataFrame
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution, Independent

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import ProgramStep
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._autocor import (
    AutocorMatrix,
    AutocorTerm,
    bind_car_terms,
    latent_residual_term,
    series_layout,
)
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas._residual_dependence import (
    ResidualLayout,
    ResidualResponse,
    TruncatedResidualRows,
    sar_weights,
)
from quivers.formulas._response_roles import bind_roles
from quivers.formulas._term import AxisDecl, Emission, SiteDecl
from quivers.formulas.family import Elliptical, Family
from quivers.formulas.formula import Formula, ResponseModifiers, _qvr_name
from quivers.inference._likelihood import row_log_likelihood
from quivers.inference._posterior import collect_site_draws
from quivers.inference._response import response_distribution
from quivers.inference.trace import Trace
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

TIME_AXIS = "FormulaAutocorTime"
"""Finite axis of the retained time levels of an unstructured correlation."""

TIME_FACTOR = "formula_cortime_factor"
"""Binding of the Cholesky factor of an unstructured correlation."""


def _residual_record(
    name: str, coefficients: Sequence[str], *, student: bool, unstructured: bool
) -> DistributionFamily:
    """Declare the checked signature of one residual response law.

    Parameters
    ----------
    name : str
        Catalog family name.
    coefficients : Sequence[str]
        Scalar structural parameter names.
    student : bool
        Include Student degrees of freedom.
    unstructured : bool
        Include the Cholesky correlation factor over time levels.

    Returns
    -------
    DistributionFamily
        Scalar-event family with one batch coordinate per response row.
    """
    parameters = [
        FamilyParameter(name="loc", constraint="real", rank=0),
        FamilyParameter(name="scale", constraint="positive", rank=0),
    ]
    if student:
        parameters.append(FamilyParameter(name="nu", constraint="positive", rank=0))
    parameters.extend(
        FamilyParameter(name=coefficient, constraint="real", rank=0)
        for coefficient in coefficients
    )
    if unstructured:
        parameters.append(
            FamilyParameter(
                name="factor",
                constraint="lower_cholesky",
                rank=2,
                scalar_broadcast=False,
            )
        )
    return DistributionFamily(
        name=name,
        parameters=tuple(parameters),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
    )


class ResidualFactory(ValueModel):
    """Build the residual response law of one retained layout.

    Parameters
    ----------
    layout : ResidualLayout
        Training or prediction row layout.
    coefficients : tuple[str, ...]
        Scalar structural parameter names.
    """

    layout: ResidualLayout = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    coefficients: tuple[str, ...] = ()

    def __call__(self, parameters: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct the law at natural-scale parameter values.

        Parameters
        ----------
        parameters : Mapping[str, torch.Tensor]
            ``loc``, ``scale``, optional ``nu``, structural coefficients, and
            an optional ``factor``.

        Returns
        -------
        Distribution
            A `ResidualResponse` over the layout's rows.
        """
        return ResidualResponse(
            self.layout,
            parameters["loc"],
            parameters["scale"],
            {name: parameters[name] for name in self.coefficients},
            parameters.get("nu"),
            parameters.get("factor"),
        )


def known_variance(
    frame: pd.DataFrame, source: str | int | float | None
) -> np.ndarray | None:
    """Read squared known standard errors of the rows of one frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Training or prediction rows.
    source : str or int or float or None
        Authored ``se()`` source, or None when the formula has none.

    Returns
    -------
    np.ndarray or None
        One squared standard error per row, or None.

    Raises
    ------
    ValueError
        If the standard errors are absent, nonnumeric, or negative.
    """
    if source is None:
        return None
    _, bound = bind_roles(frame, {"standard_error": source, "se_sigma": True})
    error = bound.get("standard_error")
    if error is None:
        raise ValueError("se() bound no standard errors")
    return np.square(np.asarray(error.column.data, dtype=np.float64))


def _data_matrix(
    term: AutocorTerm, data2: Mapping[str, AutocorMatrix] | None, rows: int, caller: str
) -> np.ndarray:
    """Read and check the ``data2`` matrix named by a SAR or FCOR term.

    Parameters
    ----------
    term : AutocorTerm
        Term naming the matrix.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    rows : int
        Included response rows.
    caller : str
        Entry point named in errors.

    Returns
    -------
    np.ndarray
        Dense float matrix of shape ``(rows, rows)``; an FCOR vector becomes a
        diagonal matrix.

    Raises
    ------
    ValueError
        If the matrix is missing, nonnumeric, or has the wrong dimensions, or
        an FCOR matrix is not symmetric positive definite.
    """
    assert term.matrix is not None
    if data2 is None or term.matrix not in data2:
        raise ValueError(f"{caller}: {term.label} requires data2[{term.matrix!r}]")
    value = data2[term.matrix]
    matrix = np.asarray(
        value.to_numpy() if isinstance(value, pd.DataFrame) else value,
        dtype=np.float64,
    )
    if term.kind == "fcor" and matrix.ndim == 1:
        matrix = np.diag(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"'M' for {term.kind.upper()} terms must be a matrix.")
    if matrix.shape != (rows, rows):
        raise ValueError(
            f"Dimensions of 'M' for {term.kind.upper()} terms must be equal to "
            "the number of observations."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"'M' for {term.kind.upper()} terms must be finite.")
    if term.kind == "fcor":
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12 * np.abs(matrix).max()):
            raise ValueError("'M' for FCOR terms must be symmetric.")
        if np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError("'M' for FCOR terms must be positive definite.")
    return matrix


def structure_layout(
    term: AutocorTerm,
    recursion: AutocorTerm | None,
    frame: pd.DataFrame,
    *,
    student: bool,
    history: np.ndarray | None,
    time_levels: tuple[str, ...] | None,
    data2: Mapping[str, AutocorMatrix] | None,
    variance: np.ndarray | None,
    caller: str,
) -> tuple[ResidualLayout, tuple[str, ...]]:
    """Code the rows of one frame for a residual term or admitted pair.

    Parameters
    ----------
    term : AutocorTerm
        Term selecting the residual law.
    recursion : AutocorTerm or None
        Conditional ARMA term combined with a SAR or fixed covariance.
    frame : pandas.DataFrame
        Included rows in original order, with a fresh range index.
    student : bool
        Student innovations.
    history : np.ndarray or None
        Response history of a conditional ARMA recursion.
    time_levels : tuple[str, ...] or None
        Retained unstructured time levels, or None to derive them.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices over the rows of ``frame``.
    variance : np.ndarray or None
        Squared known standard errors of a time-series covariance.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[ResidualLayout, tuple[str, ...]]
        Layout and the unstructured time levels.

    Raises
    ------
    ValueError
        If the rows violate a term's time, group, or matrix contract.
    """
    rows = len(frame)
    if term.dimension == "time":
        layout, levels = series_layout(
            term,
            frame,
            student=student,
            history=history,
            time_levels=time_levels,
            caller=caller,
        )
        if variance is not None:
            layout = layout.with_(known_variance=variance)
        return layout, levels
    matrix = _data_matrix(term, data2, rows, caller)
    order, starts, lengths = tuple(range(rows)), (0,), (rows,)
    if term.kind == "fcor":
        layout = ResidualLayout(
            kind="fcor",
            rows=rows,
            order=order,
            starts=starts,
            lengths=lengths,
            student=student,
            matrix=matrix,
            matrix_factor=np.linalg.cholesky(matrix),
        )
    else:
        weights, eigenvalues = sar_weights(matrix)
        layout = ResidualLayout(
            kind=term.residual_kind,
            rows=rows,
            order=order,
            starts=starts,
            lengths=lengths,
            student=student,
            weights=weights,
            eigenvalues=eigenvalues,
        )
    if recursion is not None:
        series, _ = series_layout(
            recursion,
            frame,
            student=student,
            history=None,
            time_levels=None,
            caller=caller,
        )
        layout = layout.with_(
            p=recursion.p,
            q=recursion.q,
            mean_order=series.order,
            mean_starts=series.starts,
            mean_lengths=series.lengths,
            history=history,
        )
    return layout, ()


def _structural_sites(
    term: AutocorTerm, recursion: AutocorTerm | None, layout: ResidualLayout
) -> tuple[SiteDecl, ...]:
    """Declare the scalar structural sites of a residual term or pair.

    Parameters
    ----------
    term : AutocorTerm
        Term selecting the residual law.
    recursion : AutocorTerm or None
        Conditional ARMA term combined with it.
    layout : ResidualLayout
        Training layout, whose SAR eigenvalues bound the SAR coefficient.

    Returns
    -------
    tuple[SiteDecl, ...]
        One site per coefficient, in the order of the native signature, with
        its brms class, lag, bounds, and default prior.
    """
    sites: list[SiteDecl] = []
    for item in (recursion, term):
        if item is None:
            continue
        if item.kind == "arma":
            for role, order in (("ar", item.p), ("ma", item.q)):
                for lag in range(1, order + 1):
                    bounded = item.cov
                    sites.append(
                        SiteDecl(
                            name=f"{role}_{lag}",
                            class_=role,
                            coef=str(lag),
                            default="Uniform(-1.0, 1.0)"
                            if bounded
                            else "Normal(0.0, 1.0)",
                            lower=-1.0 if bounded else None,
                            upper=1.0 if bounded else None,
                        )
                    )
        elif item.kind == "cosy":
            sites.append(
                SiteDecl(
                    name="cosy",
                    class_="cosy",
                    default="Uniform(0.0, 1.0)",
                    lower=0.0,
                    upper=1.0,
                )
            )
        elif item.kind == "sar":
            name = "lagsar" if item.sar_type == "lag" else "errorsar"
            assert layout.eigenvalues is not None
            low, high = float(layout.eigenvalues[0]), float(layout.eigenvalues[-1])
            sites.append(
                SiteDecl(
                    name=name,
                    class_=name,
                    default=f"Uniform({low!r}, {high!r})",
                    lower=low,
                    upper=high,
                )
            )
    return tuple(sites)


class ResidualLaw(ValueModel):
    """Natural residuals of one response, correlated across its rows.

    The response is ``y = mu + sigma * e`` under the family's elliptical law,
    and ``e`` has the dependence of `term`: a stationary time-series
    covariance within each series, a simultaneous autoregression, a fixed
    covariance, or a conditional ARMA recursion that moves the location by
    the lagged residuals of the recorded responses.

    Parameters
    ----------
    elliptical : Elliptical
        Residual law of the family: its scale and Student degrees of freedom.
    term : AutocorTerm
        Term selecting the dependence: a covariance, SAR, or fixed covariance
        term, or the conditional ARMA term alone.
    recursion : AutocorTerm or None
        Conditional ARMA term combined with a SAR or fixed covariance.
    se : str or int or float or None
        Authored source of known standard errors added to a time-series
        covariance, read again on other rows.
    layout : ResidualLayout
        Row coding of the rows this law covers.
    record : DistributionFamily
        Checked signature of the native kernel.
    sites : tuple[SiteDecl, ...]
        Scalar structural sites, in the order of the signature.
    time_levels : tuple[str, ...]
        Unstructured correlations only: training time levels.
    response : str
        Response column, read as recursion history on other rows.
    catalog : FamilyCatalog
        Compilation-local catalog registering the kernel over `layout`.
    caller : str
        Entry point named in refusals.
    """

    elliptical: Elliptical
    term: AutocorTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    recursion: AutocorTerm | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    se: str | int | float | None = None
    layout: ResidualLayout = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    record: DistributionFamily = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sites: tuple[SiteDecl, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    time_levels: tuple[str, ...] = ()
    response: str = ""
    catalog: FamilyCatalog = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    caller: str = ""

    @property
    def name(self) -> str:
        """The native kernel name.

        Returns
        -------
        str
            ``ResidualNormal_<y>`` or ``ResidualStudentT_<y>``.
        """
        return self.record.name

    @property
    def label(self) -> str:
        """The authored terms of the law.

        Returns
        -------
        str
            The terms joined by ``+``.
        """
        terms = (
            (self.recursion, self.term) if self.recursion is not None else (self.term,)
        )
        return " + ".join(item.label for item in terms)

    @property
    def coefficients(self) -> tuple[str, ...]:
        """The scalar structural parameter names.

        Returns
        -------
        tuple[str, ...]
            Names of `sites`.
        """
        return tuple(site.name for site in self.sites)

    @property
    def unstructured(self) -> bool:
        """Whether the law estimates a correlation over time levels.

        Returns
        -------
        bool
            True for ``unstr()``.
        """
        return self.term.kind == "unstr"

    @property
    def factorized(self) -> bool:
        """Whether the rows are independent given the parameters.

        Independent rows and a conditional ARMA recursion alone factorize:
        each row's density conditions on the recorded responses of earlier
        rows.

        Returns
        -------
        bool
            True when the pointwise score of a row is its own factor of the
            likelihood.
        """
        return self.layout.factorized

    @property
    def r_squared(self) -> Literal["residual"]:
        """The variance ``bayes_r2`` reads by default.

        The model form divides by the variance of each row's law given the
        parameters. Under correlated residuals that is not the variance of the
        row's residual about the predicted mean (a conditional recursion moves
        the mean by recorded residuals; a covariance couples the rows), so the
        residual form is the default.

        Returns
        -------
        {"residual"}
            The residual variance.
        """
        return "residual"

    @property
    def conditional(self) -> bool:
        """Whether a recursion on recorded responses moves the location.

        Returns
        -------
        bool
            True for a conditional ARMA term, alone or beside a SAR or fixed
            covariance.
        """
        return self.recursion is not None or (
            self.term.kind == "arma" and not self.term.cov
        )

    def arguments(self, location: str, scale: str) -> tuple[str, ...]:
        """Name the bindings passed to the kernel, in signature order.

        Parameters
        ----------
        location : str
            Binding of the location.
        scale : str
            Binding of the residual scale, or of the total scale under known
            errors.

        Returns
        -------
        tuple[str, ...]
            One binding per parameter of `record`.
        """
        bound = {"loc": location, "scale": scale, "factor": TIME_FACTOR}
        if self.elliptical.df is not None:
            bound["nu"] = self.elliptical.df
        bound.update((name, name) for name in self.coefficients)
        return tuple(bound[parameter.name] for parameter in self.record.parameters)

    def axes(self) -> tuple[AxisDecl, ...]:
        """Declare the finite axes the law's structural sites range over.

        Returns
        -------
        tuple[AxisDecl, ...]
            The time-level axis of an unstructured correlation, or nothing.
        """
        if not self.unstructured:
            return ()
        return (AxisDecl(name=TIME_AXIS, size=len(self.time_levels)),)

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Emit the structural sites that are not scalar.

        Parameters
        ----------
        emission : Emission
            The program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            The draw and binding of an unstructured correlation factor.
        """
        return tuple(
            step
            for axis in self.axes()
            for step in emission.correlation("Lcortime", axis, TIME_FACTOR)
        )

    def matrix_targets(self) -> tuple[tuple[str, str], ...]:
        """List the correlation-matrix prior sites.

        Returns
        -------
        tuple[tuple[str, str], ...]
            ``(site, class)`` pairs: the unstructured ``Lcortime`` factor.
        """
        return (("Lcortime", "Lcortime"),) if self.unstructured else ()

    def over(self, layout: ResidualLayout) -> ResidualLaw:
        """Return the law over other rows coded by ``layout``.

        Parameters
        ----------
        layout : ResidualLayout
            Row coding of the other rows.

        Returns
        -------
        ResidualLaw
            The same signature and sites with a catalog over ``layout``.
        """
        return self.with_(
            layout=layout, catalog=_catalog(self.record, layout, self.coefficients)
        )

    def at(
        self,
        frame: pd.DataFrame | None,
        *,
        include: bool,
        data2: Mapping[str, AutocorMatrix] | None,
        history: np.ndarray | None,
    ) -> ResidualLaw:
        """Rebuild the law for other rows, as brms's prediction does.

        New rows form their own series: ARMA history comes from the response
        column of the new rows when present (NaN rows are predicted), never
        from the training rows. A SAR or fixed-covariance term reads its
        matrix over the new rows from ``data2`` (brms ``newdata2``), and known
        standard errors are read from the new rows. ``include=False`` drops
        the dependence, leaving independent rows (brms ``incl_autocor =
        FALSE``) whose scale keeps any known standard errors.

        Parameters
        ----------
        frame : pandas.DataFrame or None
            Prediction rows, or None for the training rows.
        include : bool
            Include the dependence.
        data2 : Mapping[str, AutocorMatrix] or None
            Named matrices over the prediction rows.
        history : np.ndarray or None
            Recorded responses of the prediction rows, NaN where absent;
            read by a conditional recursion only.

        Returns
        -------
        ResidualLaw
            The law over the requested rows.

        Raises
        ------
        ValueError
            If new rows lack a SAR or FCOR matrix of their own, or violate a
            term's time, group, or matrix contract.
        """
        if frame is None and include:
            return self
        rows = self.layout.rows if frame is None else len(frame)
        fresh = None if frame is None else frame.reset_index(drop=True)
        variance = (
            self.layout.known_variance
            if fresh is None
            else known_variance(fresh, self.se)
        )
        if not include:
            return self.over(
                ResidualLayout(
                    kind="independent",
                    rows=rows,
                    order=tuple(range(rows)),
                    starts=(0,),
                    lengths=(rows,),
                    student=self.layout.student,
                    known_variance=variance,
                )
            )
        assert fresh is not None
        if self.term.dimension != "time" and (
            data2 is None or self.term.matrix not in data2
        ):
            raise ValueError(
                f"{self.term.label}: prediction on new_data requires "
                f"data2[{self.term.matrix!r}] over the new rows; predict on the "
                "training rows or pass incl_autocor=False"
            )
        layout, _ = structure_layout(
            self.term,
            self.recursion,
            fresh,
            student=self.layout.student,
            history=history if self.conditional else None,
            time_levels=self.time_levels,
            data2=data2,
            variance=variance,
            caller="new_data",
        )
        return self.over(layout)


def recorded_history(
    frame: pd.DataFrame,
    response: Callable[[pd.DataFrame], np.ndarray],
    fill: float,
    column: str,
) -> np.ndarray:
    """Read the recorded responses of prediction rows, NaN where absent.

    Parameters
    ----------
    frame : pandas.DataFrame
        Prediction rows.
    response : Callable[[pandas.DataFrame], np.ndarray]
        The fitted response transform evaluated on a frame.
    fill : float
        Value standing in for a missing response while the transform is
        evaluated; those rows are NaN in the result.
    column : str
        Response column.

    Returns
    -------
    np.ndarray
        Recorded responses in row order.
    """
    rows = len(frame)
    if column not in frame.columns:
        return np.full(rows, np.nan)
    filled = frame.copy()
    missing = filled[column].isna().to_numpy()
    if missing.any():
        filled[column] = filled[column].fillna(fill)
    values = np.array(response(filled), dtype=np.float64, copy=True).reshape(rows)
    values[missing] = np.nan
    return values


def _catalog(
    record: DistributionFamily, layout: ResidualLayout, coefficients: tuple[str, ...]
) -> FamilyCatalog:
    """Register a residual law over one layout in a fresh catalog.

    Parameters
    ----------
    record : DistributionFamily
        Checked signature.
    layout : ResidualLayout
        Rows the law covers.
    coefficients : tuple[str, ...]
        Scalar structural parameter names.

    Returns
    -------
    FamilyCatalog
        Compilation-local catalog.
    """
    return FamilyCatalog().with_family(
        record, ResidualFactory(layout=layout, coefficients=coefficients)
    )


def residual_law(
    terms: tuple[AutocorTerm, ...],
    family: Family,
    *,
    data: IntoDataFrame,
    response: np.ndarray,
    response_name: str,
    data2: Mapping[str, AutocorMatrix] | None,
    distributional: Mapping[str, str | NonlinearExpression] | None,
    modifiers: ResponseModifiers,
    caller: str,
) -> ResidualLaw:
    """Bind residual terms to the elliptical residual law of a family.

    Parameters
    ----------
    terms : tuple[AutocorTerm, ...]
        One residual term, or a conditional ARMA term with a ``sar`` or
        ``fcor`` term.
    family : Family
        Response family declaring `Family.elliptical`.
    data : IntoDataFrame
        Included training rows in original order.
    response : np.ndarray
        Evaluated response values on those rows.
    response_name : str
        Response column name.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices for ``sar`` and ``fcor``.
    distributional : Mapping[str, str or NonlinearExpression] or None
        Auxiliary-parameter formulas.
    modifiers : ResponseModifiers
        Response-side roles of the formula.
    caller : str
        Entry point named in errors.

    Returns
    -------
    ResidualLaw
        The law over the training rows.

    Raises
    ------
    ValueError
        If the family, terms, auxiliary regressions, or response roles fall
        outside the brms formulations.
    """
    conditional = next(
        (term for term in terms if term.kind == "arma" and not term.cov), None
    )
    structural = [term for term in terms if term is not conditional]
    if len(structural) > 1:
        raise ValueError("Can only model one covariance matrix of natural residuals.")
    term = structural[0] if structural else conditional
    assert term is not None
    recursion = conditional if structural else None
    label = " + ".join(item.label for item in terms)
    elliptical = family.elliptical
    if elliptical is None:
        raise ValueError(
            f"{caller}: {label} requires a family declaring an elliptical "
            f"residual law (Family.elliptical), such as 'gaussian' or 'student_t'; "
            f"{family.name!r} declares none"
        )
    if not family.location_link.is_identity:
        raise ValueError(
            f"{caller}: {label} requires the identity link of the location"
        )
    regressed = set(distributional or {})
    if elliptical.df in regressed:
        raise ValueError(f"{caller}: {label} requires a scalar {elliptical.df!r}")
    if elliptical.scale in regressed and term.kind == "sar":
        raise ValueError(
            "SAR models are not implemented when predicting "
            f"{elliptical.scale!r} or 'nu'."
        )
    # A time-series covariance takes known errors on its diagonal; the
    # modifiers of the response refuse them beside any other structure.
    error = modifiers.standard_error
    conditional = recursion is not None or (term.kind == "arma" and not term.cov)
    known = (
        error is not None
        and error.sigma
        and not conditional
        and term.kind not in ("fcor", "sar")
    )
    student = elliptical.df is not None
    frame = nw.from_native(data, eager_only=True).to_pandas().reset_index(drop=True)
    rows = len(frame)
    history = np.asarray(response, dtype=np.float64).reshape(rows)
    se_source = dict(modifiers.sources).get("standard_error") if known else None
    if isinstance(se_source, bool):
        raise ValueError(f"{caller}: se() requires a column or a numeric value")
    layout, time_levels = structure_layout(
        term,
        recursion,
        frame,
        student=student,
        history=history,
        time_levels=None,
        data2=data2,
        variance=known_variance(frame, se_source),
        caller=caller,
    )
    sites = _structural_sites(term, recursion, layout)
    coefficients = tuple(site.name for site in sites)
    clash = regressed & set(coefficients)
    if clash:
        raise ValueError(
            f"{caller}: autocorrelation parameters {sorted(clash)} cannot be regressed"
        )
    # One kernel per response, so two responses of a joint model keep their
    # own row layouts in one catalog.
    name = (
        f"{'ResidualStudentT' if student else 'ResidualNormal'}_"
        f"{_qvr_name(response_name)}"
    )
    record = _residual_record(
        name, coefficients, student=student, unstructured=term.kind == "unstr"
    )
    return ResidualLaw(
        elliptical=elliptical,
        term=term,
        recursion=recursion,
        se=se_source,
        layout=layout,
        record=record,
        sites=sites,
        time_levels=time_levels,
        response=response_name,
        catalog=_catalog(record, layout, coefficients),
        caller=caller,
    )


def bind_autocor_terms(
    terms: tuple[AutocorTerm, ...],
    family: Family,
    formula: Formula,
    *,
    data: IntoDataFrame,
    data2: Mapping[str, AutocorMatrix] | None,
    distributional: Mapping[str, str | NonlinearExpression] | None,
    nonlinear: bool,
    caller: str,
) -> tuple[ResidualLaw | None, Formula]:
    """Bind every parsed autocorrelation term of one formula.

    A residual term of a family with natural residuals binds the response's
    `ResidualLaw`; for any other family it is a latent residual term of the
    location predictor (`latent_residual_term`). A ``car`` term adds a latent
    spatial effect to the location predictor.

    Parameters
    ----------
    terms : tuple[AutocorTerm, ...]
        Parsed terms.
    family : Family
        Response family.
    formula : Formula
        Evaluated location formula.
    data : IntoDataFrame
        Included training rows.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    distributional : Mapping or None
        Auxiliary-parameter formulas.
    nonlinear : bool
        Whether the location is a nonlinear formula.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[ResidualLaw or None, Formula]
        The residual law, if any, and the formula with its latent terms.

    Raises
    ------
    ValueError
        If a term is incompatible with the family or formula, as in brms.
    """
    residual = tuple(term for term in terms if term.kind != "car")
    if family.mixture is not None:
        if residual:
            raise ValueError(
                "Autocorrelation terms cannot be applied in mixture models."
            )
        # The component locations hold the spatial fields.
        return None, formula
    if nonlinear and any(term.kind == "car" for term in terms):
        raise ValueError("CAR terms are not implemented for non-linear models.")
    if residual and nonlinear and family.elliptical is None:
        raise ValueError("Latent residuals are not implemented for non-linear models.")
    if residual and family.elliptical is None:
        # A SAR or fixed-covariance term raises brms's refusal before any
        # second latent term could be added.
        ordered = sorted(residual, key=lambda term: term.dimension == "time")
        formula = formula.with_(
            latent_terms=(
                *formula.latent_terms,
                *(
                    latent_residual_term(term, data=data, caller=caller)
                    for term in ordered
                ),
            )
        )
        residual = ()
    law = None
    if residual:
        law = residual_law(
            residual,
            family,
            data=data,
            response=formula.response_values,
            response_name=formula.response_name,
            data2=data2,
            distributional=distributional,
            modifiers=formula.modifiers,
            caller=caller,
        )
        if law.se is not None:
            # The squared errors live in the residual layout, on the diagonal
            # of each series covariance, so no scale role reaches the program.
            formula = formula.with_(
                modifiers=formula.modifiers.with_(
                    standard_error=None,
                    sources=tuple(
                        (role, source)
                        for role, source in formula.modifiers.sources
                        if role not in {"standard_error", "se_sigma"}
                    ),
                )
            )
    return law, bind_car_terms(terms, formula, data=data, data2=data2, caller=caller)


def residual_response(law: Distribution) -> ResidualResponse:
    """Unwrap event reinterpretations around a replayed residual law.

    Parameters
    ----------
    law : Distribution
        Replayed response distribution.

    Returns
    -------
    ResidualResponse
        The residual law.

    Raises
    ------
    TypeError
        If the response is not a residual dependence law.
    """
    while isinstance(law, Independent):
        law = law.base_dist
    if isinstance(law, TruncatedResidualRows):
        law = law.residual
    if not isinstance(law, ResidualResponse):
        raise TypeError("the response is not a residual dependence law")
    return law


def pointwise_scores(
    traces: Iterable[Trace],
    observations: set[str],
    residual: set[str],
    shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Collect the pointwise log likelihood of each observation.

    A law whose rows are independent given the parameters scores each row by
    its own factor of the likelihood. A residual law that correlates rows has
    no such factors, and its pointwise score is brms's leave-one-out
    conditional of each row given the others [@burkner2021efficient]. Which
    applies is read from the replayed law, so a residual law over independent
    rows (``incl_autocor = FALSE``) scores its rows' own factors.

    Parameters
    ----------
    traces : Iterable[Trace]
        Replayed executions, one per chain and draw.
    observations : set[str]
        Observation site names.
    residual : set[str]
        The observations whose law is a residual law.
    shape : tuple[int, int]
        Chain and draw counts.

    Returns
    -------
    dict[str, torch.Tensor]
        Scores of shape ``(chain, draw, ...)`` by observation.
    """

    def values() -> Iterator[dict[str, torch.Tensor]]:
        for recorded in traces:
            dependent: dict[str, ResidualResponse] = {}
            for name in residual & observations:
                law = residual_response(response_distribution(recorded, name))
                if not law.layout.factorized:
                    dependent[name] = law
            scores = row_log_likelihood(recorded, observations - dependent.keys())
            for name, law in dependent.items():
                value = recorded.sites[name].value
                scores[name] = law.loo_log_prob(value.reshape(law.loc.shape))
            yield scores

    return collect_site_draws(values(), shape)


def linear_predictor(recorded: Trace, response: str) -> torch.Tensor:
    """Read brms's ``mu`` including conditional ARMA terms from a replay.

    Parameters
    ----------
    recorded : Trace
        One replayed execution.
    response : str
        Response site name.

    Returns
    -------
    torch.Tensor
        Linear predictor of each row.
    """
    return residual_response(response_distribution(recorded, response)).linear_predictor


__all__ = ["ResidualLaw"]
