"""Residual dependence laws over the native Gaussian and Student kernels.

One `ResidualResponse` scores every brms residual structure that keeps the
response on its natural residual scale: the conditional ARMA formulation, the
stationary ARMA(1,1), compound-symmetry, unstructured and fixed covariance
formulations, and lag and error simultaneous autoregressions. Its batch axis
holds the response rows in their original order; the retained
`ResidualLayout` records which rows form one correlated series, so a
Resp-plated observation passes the whole series to one law.

`log_prob` returns one value per row whose sum is the exact joint log
density: the one-step conditional density of a conditional ARMA row, the
chain-rule factor ``log p(y_i | y_<i)`` of a covariance structure, and for a
simultaneous autoregression the row's innovation density plus an equal share
of the Jacobian. `loo_log_prob` instead returns the brms pointwise likelihood,
which for non-factorized structures is the leave-one-out predictive density
``log p(y_i | y_-i)`` [@burkner2021efficient].

Three brms combinations extend the single structures. Known standard errors
(``se(..., sigma = TRUE)``) add their squares to the diagonal of a
time-series covariance. A conditional ARMA recursion gives the location of
a simultaneous autoregression or a fixed covariance. A conditional ARMA row,
whose density factorizes, scores censored and truncated events through the
shared recorded-event masses.
"""

from __future__ import annotations

from collections.abc import Mapping
import functools
import math
from typing import Literal

import didactic.api as dx
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
import torch
from torch.types import _size
from torch.distributions import (
    Chi2,
    Distribution,
    Independent,
    MultivariateNormal,
    Normal,
    StudentT,
    TransformedDistribution,
    Transform,
    constraints,
)

from quivers._positions import anywhere, demand
from quivers._values import ValueModel, opaque_by_value
from quivers.continuous._observation_events import event_log_prob
from quivers.continuous.measure import Normalize, Restrict
from quivers.continuous._multivariate_student_t import MultivariateStudentT
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._group_covariance import _matrix_values, _tolerance
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

ResidualKind = Literal[
    "independent", "arma", "cosy", "unstr", "fcor", "sar_lag", "sar_error"
]
"""Residual structures scored on the natural response scale."""


def _vector(value: torch.Tensor, name: str) -> None:
    """Require a finite, nonempty real vector with optional leading batches.

    Parameters
    ----------
    value : torch.Tensor
        Candidate vector, whose trailing axis indexes response rows.
    name : str
        Role named in the error message.

    Raises
    ------
    ValueError
        If the value is not a tensor, has no nonempty trailing axis, or holds
        nonfinite or non-floating values.
    """
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or not value.shape[-1]:
        raise ValueError(f"{name} requires a nonempty vector")
    if not value.is_floating_point():
        raise ValueError(f"{name} requires finite floating values")
    demand(
        torch.isfinite(value),
        lambda: ValueError(f"{name} requires finite floating values"),
    )


def _matrix_tuple(
    value: tuple[tuple[float, ...], ...] | np.ndarray | torch.Tensor,
) -> tuple[tuple[float, ...], ...]:
    """Copy a known matrix into immutable nested tuples once.

    Parameters
    ----------
    value : tuple, np.ndarray, or torch.Tensor
        Known covariance or weight matrix, or an empty tuple.

    Returns
    -------
    tuple[tuple[float, ...], ...]
        Immutable Double-precision rows.
    """
    if isinstance(value, tuple) and not value:
        return ()
    return _matrix_values(value)


def _frozen_array(value: np.ndarray | None) -> np.ndarray | None:
    """Copy an optional numeric array once and clear its writeable flag.

    Parameters
    ----------
    value : np.ndarray or None
        Known data array.

    Returns
    -------
    np.ndarray or None
        A read-only Double-precision copy, or None.
    """
    if value is None:
        return None
    frozen = np.array(value, dtype=np.float64, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def arma_factor(
    scale: torch.Tensor,
    ar: torch.Tensor,
    ma: torch.Tensor,
    *,
    stationary: bool = False,
) -> torch.Tensor:
    """Factor one ARMA residual covariance without changing its innovation scale.

    Parameters
    ----------
    scale : torch.Tensor
        Positive innovation scales, with one trailing coordinate per row.
    ar, ma : torch.Tensor
        Autoregressive and moving-average coefficients, in lag order, with
        optional leading batches. Zero-length vectors omit the corresponding
        operator. Coefficients remain unconstrained for finite conditional
        initialization. Stationary initialization requires order at most one
        and an AR coefficient strictly between minus one and one.
    stationary : bool
        Use the stationary ARMA(1,1) covariance. False conditions every
        pre-series residual and innovation to zero, retaining arbitrary order.

    Returns
    -------
    torch.Tensor
        A lower-triangular factor, preserving every broadcast parameter batch.

    Raises
    ------
    ValueError
        If a scale is nonpositive or nonfinite, a coefficient vector is not
        finite, or the stationary covariance is requested outside its domain.
    """
    _vector(scale, "residual scale")
    demand(scale > 0, lambda: ValueError("residual scales must be positive"))
    for name, value in (("ar", ar), ("ma", ma)):
        if value.ndim < 1 or not value.is_floating_point():
            raise ValueError(f"{name} requires a finite coefficient vector")
        demand(
            torch.isfinite(value),
            lambda: ValueError(f"{name} requires a finite coefficient vector"),
        )
    count = scale.shape[-1]
    batch = torch.broadcast_shapes(scale.shape[:-1], ar.shape[:-1], ma.shape[:-1])
    if stationary:
        if ar.shape[-1] > 1 or ma.shape[-1] > 1:
            raise ValueError("stationary ARMA covariance requires orders at most one")
        phi = ar[..., 0] if ar.shape[-1] else scale.new_zeros(batch)
        theta = ma[..., 0] if ma.shape[-1] else scale.new_zeros(batch)
        correlation_factor = torch.linalg.cholesky(
            stationary_arma_covariance(phi, theta, count)
        )
        return scale[..., :, None] * correlation_factor
    identity = torch.eye(count, dtype=scale.dtype, device=scale.device).expand(
        *batch, count, count
    )
    autoregression = identity
    moving_average = identity
    for lag in range(1, min(count, ar.shape[-1] + 1)):
        band = torch.diag(scale.new_ones(count - lag), diagonal=-lag)
        autoregression = autoregression - ar[..., lag - 1, None, None] * band
    for lag in range(1, min(count, ma.shape[-1] + 1)):
        band = torch.diag(scale.new_ones(count - lag), diagonal=-lag)
        moving_average = moving_average + ma[..., lag - 1, None, None] * band
    innovation_factor = moving_average * scale[..., None, :]
    return torch.linalg.solve_triangular(
        autoregression, innovation_factor, upper=False, unitriangular=True
    )


def stationary_arma_covariance(
    phi: torch.Tensor, theta: torch.Tensor, count: int
) -> torch.Tensor:
    """Give the stationary ARMA(1,1) covariance per unit innovation variance.

    The diagonal is ``(1 + theta^2 + 2 phi theta) / (1 - phi^2)`` and lag
    ``k >= 1`` holds ``(1 + phi theta)(phi + theta) phi^(k-1) / (1 - phi^2)``,
    the parameterization of brms's ``cholesky_cor_arma1``. AR(1) and MA(1)
    are the cases ``theta = 0`` and ``phi = 0``.

    Parameters
    ----------
    phi, theta : torch.Tensor
        Autoregressive and moving-average coefficients with equal batch shape.
    count : int
        Number of consecutive time positions.

    Returns
    -------
    torch.Tensor
        Covariances with shape ``(*batch, count, count)``.

    Raises
    ------
    ValueError
        If ``abs(phi) >= 1``, where no stationary process exists.
    """
    demand(phi.abs() < 1, lambda: ValueError("stationary ARMA requires abs(ar) < 1"))
    positions = torch.arange(count, device=phi.device)
    lag = (positions[:, None] - positions[None, :]).abs()
    variance = (1 + theta.square() + 2 * phi * theta) / (1 - phi.square())
    first = (1 + phi * theta) * (phi + theta) / (1 - phi.square())
    powers = [torch.ones_like(phi)]
    for _ in range(max(0, count - 2)):
        powers.append(powers[-1] * phi)
    decay = torch.stack(powers, dim=-1)[..., (lag - 1).clamp_min(0)]
    covariance = first[..., None, None] * decay
    return torch.where(lag == 0, variance[..., None, None], covariance)


def compound_symmetry_factor(scale: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Use a compound-symmetry correlation with its exact positive domain.

    Parameters
    ----------
    scale : torch.Tensor
        Positive residual scales, with one trailing coordinate per row.
    rho : torch.Tensor
        Common correlation, with optional leading batches. With ``n`` rows it
        must lie strictly between ``-1 / (n - 1)`` and one.

    Returns
    -------
    torch.Tensor
        A lower-triangular factor of the scaled compound-symmetry covariance.

    Raises
    ------
    ValueError
        If a scale or the correlation is invalid, or the correlation lies
        outside its positive-definite domain.
    """
    _vector(scale, "residual scale")
    count = scale.shape[-1]
    for valid in (torch.isfinite(rho), scale > 0):
        demand(
            valid,
            lambda: ValueError(
                "residual scales and correlation must be finite and valid"
            ),
        )
    if count > 1:
        demand(
            (rho > -1 / (count - 1)) & (rho < 1),
            lambda: ValueError(
                "compound symmetry correlation is outside its positive-definite domain"
            ),
        )
    identity = torch.eye(count, dtype=scale.dtype, device=scale.device)
    covariance = identity + rho[..., None, None] * (1 - identity)
    return scale[..., :, None] * torch.linalg.cholesky(covariance)


class _ARMAFilter(Transform):
    """Invert a finite innovation filter in linear time for fixed lag orders.

    The inverse solves ``A r = M e`` for innovations ``e``, where ``A`` and
    ``M`` are the lower-triangular Toeplitz autoregressive and
    moving-average operators with zero pre-series values. Because the two
    operators commute, these innovations equal brms's conditional residuals,
    whose moving-average lags include and autoregressive lags exclude the
    autoregressive term.

    Parameters
    ----------
    loc : torch.Tensor
        Residual location added after filtering, one trailing coordinate per row.
    ar, ma : torch.Tensor
        Autoregressive and moving-average coefficients, in lag order.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, loc: torch.Tensor, ar: torch.Tensor, ma: torch.Tensor) -> None:
        super().__init__(cache_size=0)
        self.loc, self.ar, self.ma = loc, ar, ma

    @property
    def sign(self) -> int:
        """Give the sign of the unit-triangular Jacobian determinant.

        Returns
        -------
        int
            Always one, since the filter is unit lower-triangular.
        """
        return 1

    def _filter(self, values: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        """Apply the innovation filter or its inverse row by row.

        Parameters
        ----------
        values : torch.Tensor
            Innovations (forward) or located residual rows (inverse).
        inverse : bool
            Recover innovations from located rows rather than filtering them.

        Returns
        -------
        torch.Tensor
            Located residual rows (forward) or innovations (inverse).
        """
        inputs = values - self.loc if inverse else values
        outputs: list[torch.Tensor] = []
        for row in range(inputs.shape[-1]):
            value = inputs[..., row]
            for lag in range(1, min(row, self.ar.shape[-1]) + 1):
                previous = inputs[..., row - lag] if inverse else outputs[row - lag]
                value = (
                    value + (-1 if inverse else 1) * self.ar[..., lag - 1] * previous
                )
            for lag in range(1, min(row, self.ma.shape[-1]) + 1):
                previous = outputs[row - lag] if inverse else inputs[..., row - lag]
                value = (
                    value + (-1 if inverse else 1) * self.ma[..., lag - 1] * previous
                )
            outputs.append(value)
        result = torch.stack(outputs, dim=-1)
        return result if inverse else result + self.loc

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Filter innovations into located residual rows.

        Parameters
        ----------
        x : torch.Tensor
            Innovations, one trailing coordinate per row.

        Returns
        -------
        torch.Tensor
            Located residual rows.
        """
        return self._filter(x, inverse=False)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Recover innovations from located residual rows.

        Parameters
        ----------
        y : torch.Tensor
            Located residual rows.

        Returns
        -------
        torch.Tensor
            Innovations, one trailing coordinate per row.
        """
        return self._filter(y, inverse=True)

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Give the zero log-determinant of the unit-triangular filter.

        Parameters
        ----------
        x : torch.Tensor
            Innovations.
        y : torch.Tensor
            Filtered rows, unused because the determinant is constant.

        Returns
        -------
        torch.Tensor
            Zeros over the batch shape of ``x``.
        """
        return torch.zeros_like(x[..., 0])


class _ARMAInnovations(TransformedDistribution):
    """Compose existing scalar innovation laws with a unit-determinant filter.

    Parameters
    ----------
    loc : torch.Tensor
        Residual location, one trailing coordinate per row.
    scale : torch.Tensor
        Positive innovation scales on the same rows.
    ar, ma : torch.Tensor
        Autoregressive and moving-average coefficients, in lag order.
    nu : torch.Tensor or None
        Student degrees of freedom for each innovation, or None for Gaussian
        innovations.

    Raises
    ------
    ValueError
        If the location or scale is not a finite, nonempty vector.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        ar: torch.Tensor,
        ma: torch.Tensor,
        nu: torch.Tensor | None = None,
    ) -> None:
        _vector(loc, "residual location")
        _vector(scale, "residual scale")
        batch = torch.broadcast_shapes(
            loc.shape[:-1],
            scale.shape[:-1],
            ar.shape[:-1],
            ma.shape[:-1],
            () if nu is None else nu.shape,
        )
        self.loc = loc.expand(*batch, loc.shape[-1])
        self.scale = scale.expand_as(self.loc)
        self.ar, self.ma, self.nu = ar, ma, nu
        innovations = (
            Normal(torch.zeros_like(self.loc), self.scale)
            if nu is None
            else StudentT(nu[..., None], torch.zeros_like(self.loc), self.scale)
        )
        super().__init__(Independent(innovations, 1), [_ARMAFilter(self.loc, ar, ma)])

    @property
    def mean(self) -> torch.Tensor:
        """Give the residual mean, undefined for Student laws with ``nu <= 1``.

        Returns
        -------
        torch.Tensor
            The location, or NaN where the Student mean does not exist.
        """
        return (
            self.loc
            if self.nu is None
            else torch.where(self.nu[..., None] > 1, self.loc, torch.nan)
        )

    @property
    def variance(self) -> torch.Tensor:
        """Give the marginal residual variance of each row.

        Returns
        -------
        torch.Tensor
            Row variances, infinite for Student laws with ``1 < nu <= 2`` and
            NaN where ``nu <= 1``.
        """
        variance = arma_factor(self.scale, self.ar, self.ma).square().sum(-1)
        if self.nu is not None:
            multiplier = self.nu / (self.nu - 2)
            variance = torch.where(
                self.nu[..., None] > 2,
                variance * multiplier[..., None],
                torch.where(self.nu[..., None] > 1, torch.inf, torch.nan),
            )
        return variance


class ResidualLayout(ValueModel):
    """Row coding and known matrices of one residual dependence structure.

    Rows are kept in their original order. ``order`` lists original rows in
    the brms sorting by grouping factor and then time; consecutive runs of
    ``lengths`` rows starting at ``starts`` (positions in that sorted order)
    form one correlated series.

    Parameters
    ----------
    kind : ResidualKind
        Residual structure.
    rows : int
        Number of response rows.
    order : tuple[int, ...]
        Original row of each sorted position.
    starts, lengths : tuple[int, ...]
        Sorted-order start and length of each series.
    time_index : tuple[int, ...]
        Unstructured correlations only: retained time level of each sorted row.
    times : int
        Unstructured correlations only: number of retained time levels.
    p, q : int
        ARMA orders.
    stationary : bool
        ARMA only: the stationary covariance formulation (brms ``cov = TRUE``)
        rather than the conditional formulation.
    student : bool
        Student innovations rather than Gaussian innovations.
    matrix : np.ndarray or None
        Fixed covariance (``fcor``) over the rows, read-only.
    matrix_factor : np.ndarray or None
        Lower Cholesky factor of ``matrix``, computed once.
    weights : np.ndarray or None
        SAR only: ``(3, nnz)`` row indices, column indices, and values of the
        spatial weight matrix's nonzero entries.
    eigenvalues : np.ndarray or None
        SAR only: real eigenvalues of the weight matrix, computed once.
    history : np.ndarray or None
        Conditional ARMA only: response values in original row order, NaN
        where a row's response is to be predicted.
    known_variance : np.ndarray or None
        Squared known standard errors in original row order, added to the
        diagonal of each time-series covariance or to the variance of
        independent rows.
    mean_order, mean_starts, mean_lengths : tuple[int, ...]
        SAR and fixed covariances only: the group-then-time row coding of a
        conditional ARMA recursion of orders ``p`` and ``q`` that gives
        the location, coded as ``order``, ``starts``, and ``lengths`` are.
    """

    kind: ResidualKind
    rows: int
    order: tuple[int, ...]
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    time_index: tuple[int, ...] = ()
    times: int = 0
    p: int = 0
    q: int = 0
    stationary: bool = False
    student: bool = False
    matrix: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_array)
    )
    matrix_factor: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_array)
    )
    weights: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_array)
    )
    eigenvalues: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_array)
    )
    history: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_array)
    )
    known_variance: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_array)
    )
    mean_order: tuple[int, ...] = ()
    mean_starts: tuple[int, ...] = ()
    mean_lengths: tuple[int, ...] = ()

    @property
    def conditional(self) -> bool:
        """Report whether a conditional ARMA recursion moves the location.

        Returns
        -------
        bool
            True for the conditional ARMA formulation and for a SAR or fixed
            covariance combined with one.
        """
        return (self.kind == "arma" and not self.stationary) or bool(self.mean_order)

    @dx.model_validator()
    def _validate(self) -> None:
        if sorted(self.order) != list(range(self.rows)) or self.rows < 1:
            raise ValueError("residual layout order must permute every response row")
        if len(self.starts) != len(self.lengths) or sum(self.lengths) != self.rows:
            raise ValueError("residual series must partition the sorted rows")
        position = 0
        for start, length in zip(self.starts, self.lengths, strict=True):
            if start != position or length < 1:
                raise ValueError("residual series must be consecutive and nonempty")
            position += length
        if self.kind == "unstr" and (
            len(self.time_index) != self.rows
            or any(not 0 <= index < self.times for index in self.time_index)
        ):
            raise ValueError("unstructured correlations require retained time levels")
        if self.kind in {"sar_lag", "sar_error"} and (
            self.weights is None or self.eigenvalues is None
        ):
            raise ValueError("simultaneous autoregression requires spatial weights")
        if self.kind == "fcor" and (self.matrix is None or self.matrix_factor is None):
            raise ValueError("fixed residual covariance requires its matrix")
        if self.known_variance is not None and (
            self.kind not in {"independent", "arma", "cosy", "unstr"}
            or (self.kind == "arma" and not self.stationary)
            or self.known_variance.shape != (self.rows,)
            or not np.isfinite(self.known_variance).all()
            or bool((self.known_variance < 0).any())
        ):
            raise ValueError(
                "known standard errors require independent rows or a time-series "
                "covariance, and one finite nonnegative variance per row"
            )
        if self.mean_order:
            if self.kind not in {"fcor", "sar_lag", "sar_error"} or not self.p + self.q:
                raise ValueError(
                    "a conditional ARMA location requires a SAR or fixed "
                    "covariance and a positive order"
                )
            if sorted(self.mean_order) != list(range(self.rows)):
                raise ValueError("conditional ARMA order must permute every row")
            position = 0
            if len(self.mean_starts) != len(self.mean_lengths):
                raise ValueError("conditional ARMA series must partition the rows")
            for start, length in zip(self.mean_starts, self.mean_lengths, strict=True):
                if start != position or length < 1:
                    raise ValueError(
                        "conditional ARMA series must be consecutive and nonempty"
                    )
                position += length
            if position != self.rows:
                raise ValueError("conditional ARMA series must partition the rows")


def spectrum_tolerance(eigenvalues: np.ndarray) -> float:
    """Bound the rounding error of a computed spectrum.

    LAPACK's QR algorithm is backward stable: the computed eigenvalues of an
    ``n`` by ``n`` matrix ``W`` are the exact eigenvalues of ``W + E`` with
    ``||E||`` of order ``n eps ||W||``, and a simple eigenvalue moves by at
    most its condition number times ``||E||``. The bound
    is ``1e3 n eps max(1, max_i |lambda_i|)``, which allows a constant and a
    condition number whose product is at most ``1e3``. A real eigenvalue thus
    carries an imaginary part or a displacement no larger than the bound, and
    which side of an exact value a computed eigenvalue falls on depends on
    the platform's arithmetic.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Computed eigenvalues, real or complex.

    Returns
    -------
    float
        The absolute error bound of each eigenvalue.
    """
    scale = max(1.0, float(np.abs(eigenvalues).max()))
    return 1e3 * float(np.finfo(np.float64).eps) * eigenvalues.size * scale


def sar_weights(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain a SAR weight matrix's nonzeros and its eigenvalues once.

    The eigenvalues make each later log-determinant
    ``log det(I - rho W) = sum_i log(1 - rho lambda_i)`` cost O(n)
    [@ord1975estimation]. The one O(n^3) eigendecomposition happens here,
    at preparation, never during scoring.

    Parameters
    ----------
    matrix : np.ndarray
        Finite square spatial weight matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The ``(3, nnz)`` row, column, and value array of nonzero entries, and
        the real eigenvalues.

    Raises
    ------
    ValueError
        If the matrix is not finite and square or has complex eigenvalues, for
        which brms's eigenvalue bounds on the SAR parameter are undefined.
    """
    weights = np.asarray(matrix, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or not weights.size:
        raise ValueError("SAR weights must be a nonempty square matrix")
    if not np.isfinite(weights).all():
        raise ValueError("SAR weights must be finite")
    eigenvalues = np.linalg.eigvals(weights)
    if np.abs(eigenvalues.imag).max() > spectrum_tolerance(eigenvalues):
        raise ValueError(
            "SAR weights must have real eigenvalues; brms bounds the SAR "
            "parameter by the smallest and largest eigenvalue"
        )
    rows, columns = np.nonzero(weights)
    nonzero = np.stack(
        (rows.astype(np.float64), columns.astype(np.float64), weights[rows, columns])
    )
    return nonzero, np.sort(eigenvalues.real)


def _sparse_tensor(
    weights: np.ndarray, rows: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build a constant sparse weight matrix from retained nonzeros.

    Parameters
    ----------
    weights : np.ndarray
        ``(3, nnz)`` rows, columns, and values.
    rows : int
        Matrix dimension.
    dtype : torch.dtype
        Floating dtype of the scored values.
    device : torch.device
        Device of the scored values.

    Returns
    -------
    torch.Tensor
        A coalesced sparse COO matrix.
    """
    index = torch.as_tensor(weights[:2], dtype=torch.long, device=device)
    values = torch.as_tensor(weights[2], dtype=dtype, device=device)
    return torch.sparse_coo_tensor(index, values, (rows, rows)).coalesce()


def _sparse_apply(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Multiply a constant sparse matrix into the trailing axis of a batch.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse ``(n, n)`` matrix.
    values : torch.Tensor
        Values with trailing axis ``n`` and arbitrary leading batches.

    Returns
    -------
    torch.Tensor
        ``matrix @ value`` for every batch element, in O(nnz) per element.
    """
    flat = values.reshape(-1, values.shape[-1]).mT
    return torch.sparse.mm(matrix, flat).mT.reshape(values.shape)


def _scalar(value: torch.Tensor, rows: int) -> torch.Tensor:
    """Read one shared parameter that a plated observation broadcast over rows.

    Parameters
    ----------
    value : torch.Tensor
        Shared value, possibly repeated along a trailing row axis.
    rows : int
        Number of response rows.

    Returns
    -------
    torch.Tensor
        The value without its repeated row axis.
    """
    return value[..., 0] if value.ndim and value.shape[-1] == rows else value


class ResidualResponse(Distribution):
    """Score, predict, and sample one residual dependence structure.

    The law has one batch coordinate per response row, in original order, and
    no event axis; a plated formula observation passes all rows at once.

    Parameters
    ----------
    layout : ResidualLayout
        Row coding and known matrices.
    loc : torch.Tensor
        Linear predictor of each row, before any autocorrelation term.
    scale : torch.Tensor
        Positive residual (innovation) scale of each row.
    coefficients : Mapping[str, torch.Tensor]
        ``ar_k`` and ``ma_k`` (ARMA), ``cosy``, ``lagsar`` or ``errorsar``,
        each possibly repeated over rows.
    nu : torch.Tensor or None
        Student degrees of freedom, or None for Gaussian innovations.
    factor : torch.Tensor or None
        Unstructured correlations only: Cholesky factor over time levels.

    Raises
    ------
    ValueError
        If parameters do not cover the layout's rows or leave their domains.

    Notes
    -----
    Costs per evaluation, with ``n`` rows, ``m`` the longest series, and
    ``s`` series: conditional ARMA O(n (p + q)); stationary ARMA and compound
    symmetry one O(m^3) Cholesky factor shared by every series plus
    O(sum of squared series lengths) triangular solves; unstructured
    correlations a separate O(n_g^3) factor only for a series missing time
    levels; fixed covariance O(n^2) with the factor computed once; SAR O(nnz)
    sparse products and an O(n) eigenvalue log-determinant. Prediction from a
    lag SAR solves ``(I - rho W) x = b`` by sparse LU. Known standard errors
    give every series its own covariance, one O(n_g^3) factor per series. A
    conditional ARMA location adds O(n (p + q)) to a SAR or fixed covariance.
    Censored and truncated conditional ARMA rows cost O(n (p + q)) plus one
    tail evaluation per row.
    """

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Declare no torch-validated arguments; domains are checked per draw.

        Returns
        -------
        dict[str, constraints.Constraint]
            An empty mapping.
        """
        return {}

    @property
    def support(self) -> constraints.Constraint:
        """Give the real response support.

        Returns
        -------
        constraints.Constraint
            The real line.
        """
        return constraints.real

    def __init__(
        self,
        layout: ResidualLayout,
        loc: torch.Tensor,
        scale: torch.Tensor,
        coefficients: Mapping[str, torch.Tensor],
        nu: torch.Tensor | None = None,
        factor: torch.Tensor | None = None,
    ) -> None:
        rows = layout.rows
        if (
            loc.ndim < 1
            or scale.ndim < 1
            or loc.shape[-1] != rows
            or scale.shape[-1] != rows
        ):
            raise ValueError(
                f"residual structure covers {rows} rows, got {tuple(loc.shape)}"
            )
        self.layout = layout
        self.loc, self.scale = torch.broadcast_tensors(loc, scale)
        # Parameters outside their domain receive zero density rather than an
        # exception, so samplers can reject the proposal.
        self._valid = (
            torch.isfinite(self.loc).all(-1)
            & torch.isfinite(self.scale).all(-1)
            & (self.scale > 0).all(-1)
        )
        self.scale = torch.where(
            self.scale > 0, self.scale, torch.ones_like(self.scale)
        )
        if layout.kind == "independent" and layout.known_variance is not None:
            known = torch.as_tensor(
                np.array(layout.known_variance),
                dtype=self.scale.dtype,
                device=self.scale.device,
            )
            self.scale = (self.scale.square() + known).sqrt()
        self.nu = None if nu is None else _scalar(nu, rows)
        if layout.student and self.nu is None:
            raise ValueError("Student residuals require degrees of freedom")
        self.coefficients = {
            name: _scalar(value, rows) for name, value in coefficients.items()
        }
        self.factor = (
            None
            if factor is None
            else factor[..., 0, :, :]
            if factor.ndim >= 3
            and factor.shape[-3] == rows
            and factor.shape[-2] == layout.times
            else factor
        )
        self._order = torch.as_tensor(layout.order, device=loc.device)
        self._inverse_order = torch.argsort(self._order)
        self._bounds: tuple[
            torch.Tensor | float | None, torch.Tensor | float | None
        ] = (None, None)
        super().__init__(self.loc.shape, validate_args=False)

    @property
    def has_rsample(self) -> bool:
        """Report reparameterized sampling for every structure except SAR.

        Returns
        -------
        bool
            False for SAR, whose draws use a sparse LU solve outside autograd.
        """
        return self.layout.kind not in {"sar_lag", "sar_error"}

    def _ar_ma(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack the per-lag ARMA coefficients.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Autoregressive and moving-average coefficients in lag order.
        """
        batch = self.loc.shape[:-1]
        empty = self.loc.new_zeros((*batch, 0))
        ar = (
            torch.stack(
                [
                    self.coefficients[f"ar_{lag}"].expand(batch)
                    for lag in range(1, self.layout.p + 1)
                ],
                -1,
            )
            if self.layout.p
            else empty
        )
        ma = (
            torch.stack(
                [
                    self.coefficients[f"ma_{lag}"].expand(batch)
                    for lag in range(1, self.layout.q + 1)
                ],
                -1,
            )
            if self.layout.q
            else empty
        )
        return ar, ma

    def _series(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Index the sorted rows as padded series.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Padded original-row indices ``(s, m)``, their validity mask, and
            the ``(series, position)`` of every original row.
        """
        layout = self.layout
        order, starts, lengths = (
            (layout.mean_order, layout.mean_starts, layout.mean_lengths)
            if layout.mean_order
            else (layout.order, layout.starts, layout.lengths)
        )
        longest = max(lengths)
        index = torch.zeros(len(lengths), longest, dtype=torch.long)
        mask = torch.zeros(len(lengths), longest, dtype=torch.bool)
        coordinates = torch.zeros(layout.rows, 2, dtype=torch.long)
        for series, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            rows = torch.as_tensor(order[start : start + length])
            index[series, :length] = rows
            mask[series, :length] = True
            coordinates[rows, 0] = series
            coordinates[rows, 1] = torch.arange(length)
        device = self.loc.device
        return index.to(device), mask.to(device), coordinates.to(device)

    def _innovation(self, location: torch.Tensor, scale: torch.Tensor) -> Distribution:
        """Build the independent scalar innovation law of each row.

        Parameters
        ----------
        location, scale : torch.Tensor
            Conditional location and scale of each row.

        Returns
        -------
        Distribution
            Normal or Student law with the same batch shape.
        """
        if self.nu is None:
            return Normal(location, scale)
        return StudentT(self.nu[..., None].expand_as(location), location, scale)

    def _arma_means(
        self,
        values: torch.Tensor,
        *,
        fill: bool,
        bounds: tuple[torch.Tensor | float | None, torch.Tensor | float | None] = (
            None,
            None,
        ),
    ) -> torch.Tensor:
        """Run brms's conditional ARMA predictor over each series.

        The conditional mean of row ``n`` adds ``sum ma_k err_(n-k)`` and
        ``sum ar_k err_(n-k)`` to the linear predictor, where
        ``err_n = y_n - eta_n - sum ma_k err_(n-k)``. A missing response is
        first drawn from its conditional law, truncated to the row's
        sampling bounds when it has them, and then used as history, as in
        brms's out-of-sample prediction.

        Parameters
        ----------
        values : torch.Tensor
            Response history in original row order, NaN where missing.
        fill : bool
            Draw missing history; False requires complete history.
        bounds : tuple
            Inclusive truncation bounds of the response sampling law, each a
            scalar, one value per row, or None for an open side.

        Returns
        -------
        torch.Tensor
            Conditional means in original row order, O(n (p + q)).

        Raises
        ------
        ValueError
            If history is missing and ``fill`` is False, or history is
            missing under a SAR or fixed covariance, whose joint law gives
            brms no pointwise draw of one missing response.
        """
        index, mask, coordinates = self._series()
        ar, ma = self._ar_ma()
        eta = self.loc[..., index]
        history = values.expand_as(self.loc)[..., index]
        scale = self.scale[..., index]
        limits = [
            None
            if bound is None
            else torch.as_tensor(
                bound, dtype=self.loc.dtype, device=self.loc.device
            ).expand(self.layout.rows)[index]
            for bound in bounds
        ]
        errors: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        for position in range(index.shape[1]):
            base = eta[..., position]
            for lag in range(1, min(position, self.layout.q) + 1):
                base = base + ma[..., lag - 1, None] * errors[position - lag]
            mean = base
            for lag in range(1, min(position, self.layout.p) + 1):
                mean = mean + ar[..., lag - 1, None] * errors[position - lag]
            current = history[..., position]
            missing = torch.isnan(current) & mask[:, position]
            if anywhere(missing):
                if not fill:
                    raise ValueError("conditional ARMA scoring requires every response")
                if self.layout.mean_order:
                    raise ValueError(
                        "conditional ARMA terms combined with a SAR or fixed "
                        "covariance require the response of every row: brms "
                        "defines no pointwise draw of a missing response under "
                        "the joint law"
                    )
                innovation = self._innovation(mean, scale[..., position])
                low, high = (
                    None if limit is None else limit[..., position] for limit in limits
                )
                draw = (
                    innovation.sample()
                    if low is None and high is None
                    else Normalize(Restrict(innovation, low, high))
                    .sample()
                    .to(mean.dtype)
                )
                current = torch.where(missing, draw, current)
            current = torch.where(mask[:, position], current, torch.zeros_like(mean))
            errors.append(current - base)
            means.append(mean)
        stacked = torch.stack(means, -1)
        return stacked[..., coordinates[:, 0], coordinates[:, 1]]

    def _history(self) -> torch.Tensor:
        """Return retained response history in original row order.

        Returns
        -------
        torch.Tensor
            History with NaN where a response is to be predicted.
        """
        history = self.layout.history
        if history is None:
            return torch.full_like(self.loc, torch.nan)
        return torch.as_tensor(
            np.array(history), dtype=self.loc.dtype, device=self.loc.device
        )

    def _correlation_factor(self, start: int, length: int) -> torch.Tensor:
        """Factor the residual correlation of one series.

        Parameters
        ----------
        start, length : int
            Sorted-order extent of the series.

        Returns
        -------
        torch.Tensor
            Lower Cholesky factor of its correlation (or fixed covariance).

        Raises
        ------
        ValueError
            If a structural parameter leaves its positive-definite domain.
        """
        layout = self.layout
        if layout.kind == "fcor":
            assert layout.matrix_factor is not None
            return torch.as_tensor(
                layout.matrix_factor, dtype=self.loc.dtype, device=self.loc.device
            )
        if layout.kind == "unstr":
            assert self.factor is not None
            levels = layout.time_index[start : start + length]
            if levels == tuple(range(layout.times)):
                return self.factor
            index = torch.as_tensor(levels, device=self.loc.device)
            correlation = self.factor @ self.factor.mT
            factor, info = torch.linalg.cholesky_ex(
                correlation[..., index, :][..., :, index]
            )
            self._valid = self._valid & (info == 0)
            return factor
        return self._shared_factor()[..., :length, :length]

    def _shared_factor(self) -> torch.Tensor:
        """Factor the longest stationary ARMA or compound-symmetry correlation.

        The leading block of a Cholesky factor is the factor of the leading
        block, so every shorter series reuses this one factor, as in brms.

        Returns
        -------
        torch.Tensor
            Lower Cholesky factor over the longest series.

        Raises
        ------
        ValueError
            If a structural parameter leaves its positive-definite domain.
        """
        cached = getattr(self, "_cached_factor", None)
        if cached is not None:
            return cached
        longest = max(self.layout.lengths)
        if self.layout.kind == "arma":
            batch = self.loc.shape[:-1]
            phi = (
                self.coefficients["ar_1"].expand(batch)
                if self.layout.p
                else self.loc.new_zeros(batch)
            )
            theta = (
                self.coefficients["ma_1"].expand(batch)
                if self.layout.q
                else self.loc.new_zeros(batch)
            )
            inside = (phi.abs() < 1) & (theta.abs() <= 1)
            self._valid = self._valid & inside
            covariance = stationary_arma_covariance(
                torch.where(inside, phi, torch.zeros_like(phi)),
                torch.where(inside, theta, torch.zeros_like(theta)),
                longest,
            )
        else:
            rho = self.coefficients["cosy"].expand(self.loc.shape[:-1])
            inside = (rho < 1) & (
                (rho > -1 / (longest - 1)) if longest > 1 else torch.ones_like(rho > 0)
            )
            self._valid = self._valid & inside
            rho = torch.where(inside, rho, torch.zeros_like(rho))
            identity = torch.eye(longest, dtype=self.loc.dtype, device=self.loc.device)
            covariance = identity + rho[..., None, None] * (1 - identity)
        factor, info = torch.linalg.cholesky_ex(covariance)
        self._valid = self._valid & (info == 0)
        self._cached_factor = factor
        return factor

    def _blocks(self):
        """Yield every correlated series with its scaled covariance factor.

        Yields
        ------
        tuple[torch.Tensor, torch.Tensor]
            Original rows of the series and the lower factor of its
            covariance, which includes any squared known standard errors.
        """
        known = self.layout.known_variance
        for start, length in zip(self.layout.starts, self.layout.lengths, strict=True):
            rows = self._order[start : start + length]
            factor = self.scale[..., rows, None] * self._correlation_factor(
                start, length
            )
            if known is not None:
                # brms adds se^2 to the covariance itself, so each series
                # needs its own factor.
                variance = torch.as_tensor(
                    np.array(known), dtype=self.loc.dtype, device=self.loc.device
                )[rows]
                factor, info = torch.linalg.cholesky_ex(
                    factor @ factor.mT + torch.diag_embed(variance)
                )
                self._valid = self._valid & (info == 0)
            yield rows, factor

    def _sar(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sparse weights, SAR parameter, and log-determinant.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Sparse ``W``, ``rho``, and ``log det(I - rho W)``.

        Raises
        ------
        ValueError
            If ``I - rho W`` is singular or has a nonpositive determinant
            factor, outside brms's eigenvalue bounds.
        """
        layout = self.layout
        assert layout.weights is not None and layout.eigenvalues is not None
        weights = _sparse_tensor(
            layout.weights, layout.rows, self.loc.dtype, self.loc.device
        )
        name = "lagsar" if layout.kind == "sar_lag" else "errorsar"
        rho = self.coefficients[name]
        eigenvalues = torch.as_tensor(
            layout.eigenvalues, dtype=self.loc.dtype, device=self.loc.device
        )
        factors = 1 - rho[..., None] * eigenvalues
        # A factor within the rounding error of its eigenvalue, computed in
        # Float64 and rounded to the scoring dtype, is not distinguishable
        # from zero, so the matrix is treated as singular.
        tolerance = max(
            spectrum_tolerance(layout.eigenvalues), torch.finfo(factors.dtype).eps
        )
        nonsingular = factors > rho.abs()[..., None] * tolerance
        self._valid = self._valid & nonsingular.all(-1)
        return (
            weights,
            rho,
            torch.where(nonsingular, factors, torch.ones_like(factors)).log().sum(-1),
        )

    def _location(self, values: torch.Tensor, *, fill: bool) -> torch.Tensor:
        """Give the location of the covariance or SAR law.

        Parameters
        ----------
        values : torch.Tensor
            Response history in original row order.
        fill : bool
            Passed to `_arma_means` for missing history.

        Returns
        -------
        torch.Tensor
            The linear predictor, with conditional ARMA terms when the layout
            combines them with a SAR or fixed covariance.
        """
        if self.layout.mean_order:
            return self._arma_means(values, fill=fill)
        return self.loc

    def _sar_residual(
        self, value: torch.Tensor, location: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map responses to SAR innovations by the inverse linear transform.

        Parameters
        ----------
        value : torch.Tensor
            Responses in original row order.
        location : torch.Tensor
            Location of each row.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Innovations ``(I - rho W) y - eta`` (lag) or
            ``(I - rho W)(y - eta)`` (error), and the log-determinant.
        """
        weights, rho, log_det = self._sar()
        if self.layout.kind == "sar_lag":
            residual = value - rho[..., None] * _sparse_apply(weights, value)
            return residual - location, log_det
        centered = value - location
        return centered - rho[..., None] * _sparse_apply(weights, centered), log_det

    def _chain_rule(self, standardized: torch.Tensor) -> torch.Tensor:
        """Split a standardized multivariate density into one-step conditionals.

        Parameters
        ----------
        standardized : torch.Tensor
            Whitened residuals of one series along the trailing axis.

        Returns
        -------
        torch.Tensor
            ``log p(z_i | z_<i)`` for each coordinate; Gaussian factors are
            standard Normal and Student factors use the conditional
            Student law with ``nu + i - 1`` degrees of freedom.
        """
        if self.nu is None:
            return Normal(0.0, 1.0).log_prob(standardized)
        nu = self.nu[..., None]
        before = torch.cumsum(standardized.square(), -1) - standardized.square()
        count = torch.arange(standardized.shape[-1], device=standardized.device)
        degrees = nu + count
        scale = ((nu + before) / degrees).sqrt()
        return StudentT(degrees, torch.zeros_like(scale), scale).log_prob(standardized)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score each row by a factor of the exact joint density.

        Parameters outside their domain (a nonpositive scale, a
        nonstationary ARMA(1,1), a compound-symmetry correlation outside its
        positive-definite interval, a singular SAR operator) give every row
        of that batch element a score of minus infinity.

        Parameters
        ----------
        value : torch.Tensor
            Responses in original row order.

        Returns
        -------
        torch.Tensor
            Row scores whose sum is the joint log density.
        """
        return self._mask(self._log_prob(value))

    def _mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Assign zero density to batch elements outside the parameter domain.

        Parameters
        ----------
        scores : torch.Tensor
            Row scores.

        Returns
        -------
        torch.Tensor
            Scores, minus infinity where a parameter left its domain.
        """
        return torch.where(
            self._valid[..., None], scores, torch.full_like(scores, -torch.inf)
        )

    def _log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score each row by a factor of the exact joint density, unmasked.

        Parameters
        ----------
        value : torch.Tensor
            Responses in original row order.

        Returns
        -------
        torch.Tensor
            Row scores whose sum is the joint log density.

        Raises
        ------
        ValueError
            If a structural parameter leaves its domain.
        """
        layout = self.layout
        value = value.to(self.loc.dtype)
        if layout.kind == "independent":
            return self._innovation(self.loc, self.scale).log_prob(value)
        if layout.kind == "arma" and not layout.stationary:
            index, mask, coordinates = self._series()
            ar, ma = self._ar_ma()
            filtered = _ARMAFilter(
                self.loc[..., index], ar[..., None, :], ma[..., None, :]
            )
            innovations = filtered._inverse(value.expand_as(self.loc)[..., index])
            rows = innovations[..., coordinates[:, 0], coordinates[:, 1]]
            return self._innovation(torch.zeros_like(rows), self.scale).log_prob(rows)
        location = self._location(value, fill=False)
        if layout.kind in {"sar_lag", "sar_error"}:
            residual, log_det = self._sar_residual(value.expand_as(self.loc), location)
            terms = self._chain_rule(residual / self.scale) - self.scale.log()
            return terms + (log_det / layout.rows)[..., None]
        scores = torch.zeros_like(self.loc)
        for rows, factor in self._blocks():
            centered = (value - location)[..., rows]
            standardized = torch.linalg.solve_triangular(
                factor, centered[..., None], upper=False
            )[..., 0]
            terms = (
                self._chain_rule(standardized)
                - torch.diagonal(factor, dim1=-2, dim2=-1).log()
            )
            scores = scores.index_copy(-1, rows, terms)
        return scores

    def loo_log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score rows as brms's ``log_lik`` does, with domain masking.

        Parameters
        ----------
        value : torch.Tensor
            Responses in original row order.

        Returns
        -------
        torch.Tensor
            Pointwise log likelihoods; see `_loo_log_prob`.
        """
        return self._mask(self._loo_log_prob(value))

    def _loo_log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score rows as brms's ``log_lik`` does.

        A conditional ARMA row keeps its one-step conditional density. A
        covariance or SAR structure uses the leave-one-out predictive density
        ``log p(y_i | y_-i)``: with precision ``C``, ``g = C e``, and
        ``c_i = C_ii``, a Gaussian row has mean ``y_i - g_i / c_i`` and
        variance ``1 / c_i``. A Student row uses ``nu + n - 1`` degrees of
        freedom and scales that variance by
        ``(nu + e_-i' C_(-i) e_-i) / (nu + n - 1)``, where
        ``e_-i' C_(-i) e_-i = e' C e - g_i^2 / c_i`` [@burkner2021efficient].

        Parameters
        ----------
        value : torch.Tensor
            Responses in original row order.

        Returns
        -------
        torch.Tensor
            Pointwise log likelihoods. Their sum is not the joint density for
            a covariance or SAR structure.
        """
        layout = self.layout
        value = value.to(self.loc.dtype)
        if layout.kind == "independent" or (
            layout.kind == "arma" and not layout.stationary
        ):
            return self.log_prob(value)
        if layout.kind in {"sar_lag", "sar_error"}:
            # C = A' S^-2 A with A = I - rho W and S = diag(scale); every
            # quantity below is a sparse product, O(nnz).
            weights, rho, _ = self._sar()
            residual, _ = self._sar_residual(
                value.expand_as(self.loc), self._location(value, fill=False)
            )
            inverse = self.scale.square().reciprocal()
            weighted = residual * inverse
            transpose = weights.transpose(0, 1).coalesce()
            gradient = weighted - rho[..., None] * _sparse_apply(transpose, weighted)
            assert layout.weights is not None
            squares = torch.sparse_coo_tensor(
                transpose.indices(), transpose.values().square(), transpose.shape
            ).coalesce()
            on_diagonal = layout.weights[0] == layout.weights[1]
            diagonal = torch.zeros(
                layout.rows, dtype=self.loc.dtype, device=self.loc.device
            ).index_add(
                0,
                torch.as_tensor(layout.weights[0][on_diagonal], dtype=torch.long),
                torch.as_tensor(layout.weights[2][on_diagonal], dtype=self.loc.dtype),
            )
            precision = inverse * (1 - 2 * rho[..., None] * diagonal) + rho[
                ..., None
            ].square() * _sparse_apply(squares, inverse)
            quadratic = (residual * weighted).sum(-1, keepdim=True)
            return self._loo(value, gradient, precision, quadratic, layout.rows)
        scores = torch.zeros_like(self.loc)
        location = self._location(value, fill=False)
        for rows, factor in self._blocks():
            centered = (value - location)[..., rows]
            precision = torch.cholesky_inverse(factor)
            gradient = (precision @ centered[..., None])[..., 0]
            quadratic = (centered * gradient).sum(-1, keepdim=True)
            terms = self._loo(
                value.expand_as(self.loc)[..., rows],
                gradient,
                torch.diagonal(precision, dim1=-2, dim2=-1),
                quadratic,
                len(rows),
            )
            scores = scores.index_copy(-1, rows, terms)
        return scores

    def _loo(
        self,
        value: torch.Tensor,
        gradient: torch.Tensor,
        precision: torch.Tensor,
        quadratic: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        """Evaluate brms's leave-one-out predictive densities of one series.

        Parameters
        ----------
        value : torch.Tensor
            Responses of the series.
        gradient : torch.Tensor
            Precision times residual, ``g = C e``.
        precision : torch.Tensor
            Diagonal of the precision, ``c``.
        quadratic : torch.Tensor
            ``e' C e``, with a trailing singleton axis.
        count : int
            Series length ``n``.

        Returns
        -------
        torch.Tensor
            Leave-one-out log densities of the series' rows.
        """
        location = value - gradient / precision
        variance = 1 / precision
        if self.nu is None:
            return Normal(location, variance.sqrt()).log_prob(value)
        nu = self.nu[..., None]
        remaining = quadratic - gradient.square() / precision
        variance = variance * (nu + remaining) / (nu + count - 1)
        degrees = (nu + count - 1).expand_as(location)
        return StudentT(degrees, location, variance.sqrt()).log_prob(value)

    @property
    def mean(self) -> torch.Tensor:
        """Give brms's expected response.

        A conditional ARMA mean adds the ARMA terms computed from retained
        response history, drawing missing history from the predictive law;
        a lag SAR mean is ``(I - rho W)^-1 mu`` with ``mu`` the linear
        predictor including any conditional ARMA terms; every other
        structure has mean ``mu``.

        Returns
        -------
        torch.Tensor
            Expected responses in original row order.
        """
        layout = self.layout
        if layout.kind == "sar_lag":
            return self._sar_solve(self.linear_predictor)
        return self.linear_predictor

    @property
    def linear_predictor(self) -> torch.Tensor:
        """Give brms's ``mu``, including conditional ARMA terms.

        Returns
        -------
        torch.Tensor
            The linear predictor with any conditional ARMA contribution.
        """
        if self.layout.conditional:
            return self._arma_means(self._history(), fill=True, bounds=self._bounds)
        return self.loc

    def _sar_solve(self, values: torch.Tensor) -> torch.Tensor:
        """Solve ``(I - rho W) x = b`` by sparse LU for each batch element.

        Parameters
        ----------
        values : torch.Tensor
            Right-hand sides in original row order.

        Returns
        -------
        torch.Tensor
            Solutions, detached from autograd.
        """
        layout = self.layout
        assert layout.weights is not None
        rows, columns, entries = layout.weights
        weights = sparse.csc_matrix(
            (entries, (rows.astype(int), columns.astype(int))),
            shape=(layout.rows, layout.rows),
        )
        name = "lagsar" if layout.kind == "sar_lag" else "errorsar"
        rho = self.coefficients[name].detach().expand(values.shape[:-1])
        flat_rho = rho.reshape(-1).cpu().numpy()
        flat = values.detach().reshape(-1, layout.rows).cpu().numpy()
        identity = sparse.identity(layout.rows, format="csc")
        solved = np.stack(
            [
                sparse_linalg.splu((identity - r * weights).tocsc()).solve(b)
                for r, b in zip(flat_rho, flat, strict=True)
            ]
        )
        return torch.as_tensor(
            solved.reshape(values.shape), dtype=values.dtype, device=values.device
        )

    def _radius(self, shape: torch.Size, series: int) -> torch.Tensor:
        """Draw one Student mixing scale per correlated series.

        Parameters
        ----------
        shape : torch.Size
            Requested sample shape.
        series : int
            Number of correlated series.

        Returns
        -------
        torch.Tensor
            ``sqrt(nu / chi2_nu)`` with shape ``(*shape, *batch, series)``.
        """
        assert self.nu is not None
        nu = self.nu[..., None].expand(*self.loc.shape[:-1], series)
        return (nu / Chi2(nu).sample(shape)).sqrt()

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw responses with brms's ``posterior_predict`` semantics.

        Conditional ARMA rows draw independently around their conditional
        means, which use retained history and separately drawn missing
        history. Covariance structures draw each series jointly; Student
        series share one mixing scale per series.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Draws with shape ``(*sample_shape, *batch_shape)``.
        """
        shape = torch.Size(sample_shape)
        layout = self.layout
        if layout.kind == "independent":
            return self._innovation(self.loc, self.scale).rsample(shape)
        if layout.kind == "arma" and not layout.stationary:
            draws = [
                self._innovation(self.linear_predictor, self.scale).rsample()
                for _ in range(math.prod(shape))
            ]
            return torch.stack(draws).reshape(*shape, *self.loc.shape)
        location = self.linear_predictor
        if layout.kind in {"sar_lag", "sar_error"}:
            noise = torch.randn((*shape, *self.loc.shape), dtype=self.loc.dtype)
            noise = noise * self.scale
            if self.nu is not None:
                noise = noise * self._radius(shape, 1)
            if layout.kind == "sar_lag":
                return self._sar_solve(location + noise)
            return location + self._sar_solve(noise)
        result = location.expand(*shape, *self.loc.shape).clone()
        radius = None if self.nu is None else self._radius(shape, len(layout.lengths))
        for series, (rows, factor) in enumerate(self._blocks()):
            noise = torch.randn(
                (*shape, *self.loc.shape[:-1], len(rows)),
                dtype=self.loc.dtype,
                device=self.loc.device,
            )
            draw = (factor @ noise[..., None])[..., 0]
            if radius is not None:
                draw = draw * radius[..., series, None]
            result = result.index_add(-1, rows, draw)
        return result

    def _factorized(self, term: str) -> None:
        """Require a structure whose rows are conditionally independent.

        Parameters
        ----------
        term : str
            Response terms named in the error.

        Raises
        ------
        ValueError
            With brms's message for a covariance, SAR, or fixed-covariance
            structure, whose likelihood brms does not adjust per row.
        """
        layout = self.layout
        if layout.kind != "independent" and (
            layout.kind != "arma" or layout.stationary
        ):
            raise ValueError(
                f"Invalid addition arguments for this model: {term} requires the "
                "conditional ARMA formulation"
            )

    def recorded_event_log_prob(
        self,
        value: torch.Tensor,
        *,
        status: torch.Tensor | int | None = None,
        interval_upper: torch.Tensor | float | None = None,
        trunc_lower: torch.Tensor | float | None = None,
        trunc_upper: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Score censored and truncated rows of a conditional ARMA law.

        brms computes the ARMA residuals from the recorded responses, a
        censoring endpoint included, and then scores each row's event under
        its conditional Normal or Student law
        ``F(mu_n + ARMA terms, sigma)``: the density of an exact row, a tail
        or interval mass of a censored row, each divided by the mass of the
        row's truncation interval.

        Parameters
        ----------
        value : torch.Tensor
            Recorded responses in original row order; the lower endpoint of
            an interval-censored row.
        status : torch.Tensor or int or None
            Codes -1 left, 0 exact, 1 right, 2 interval; None means exact.
        interval_upper : torch.Tensor or float or None
            Upper endpoints of interval-censored rows.
        trunc_lower, trunc_upper : torch.Tensor or float or None
            Inclusive sampling bounds.

        Returns
        -------
        torch.Tensor
            Row scores, O(n (p + q)) plus one tail evaluation per row.

        Raises
        ------
        ValueError
            For a covariance, SAR, or fixed-covariance structure, as in brms.
        """
        if (
            status is None
            and interval_upper is None
            and trunc_lower is None
            and trunc_upper is None
        ):
            return self.log_prob(value)
        self._factorized("cens() and trunc()")
        value = value.to(self.loc.dtype)
        location = (
            self._arma_means(value, fill=False) if self.layout.conditional else self.loc
        )
        return self._mask(
            event_log_prob(
                self._innovation(location, self.scale),
                value,
                status=status,
                interval_upper=interval_upper,
                trunc_lower=trunc_lower,
                trunc_upper=trunc_upper,
            )
        )

    def componentwise_truncation(
        self, lower: torch.Tensor | float | None, upper: torch.Tensor | float | None
    ) -> Distribution:
        """Give the truncated sampling law of conditional ARMA rows.

        Missing response history is drawn from the truncated conditional law
        before it enters later rows, as brms's prediction draws it.

        Parameters
        ----------
        lower, upper : torch.Tensor or float or None
            Inclusive truncation bounds; None leaves a side open.

        Returns
        -------
        Distribution
            Independent rows of the conditional law, each normalized on its
            truncation interval.

        Raises
        ------
        ValueError
            For a covariance, SAR, or fixed-covariance structure, as in brms.
        """
        self._factorized("trunc()")
        self._bounds = (lower, upper)
        return TruncatedResidualRows(self, lower, upper)

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw responses without gradients.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Detached draws.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)


class TruncatedResidualRows(Normalize):
    """Conditional ARMA rows normalized on their truncation intervals.

    Parameters
    ----------
    law : ResidualResponse
        Conditional ARMA or independent residual law, retained so that the
        linear predictor stays readable from the truncated sampling law.
    lower, upper : torch.Tensor or float or None
        Inclusive truncation bounds; None leaves a side open.
    """

    def __init__(
        self,
        law: ResidualResponse,
        lower: torch.Tensor | float | None,
        upper: torch.Tensor | float | None,
    ) -> None:
        self.residual = law
        super().__init__(
            Restrict(law._innovation(law.linear_predictor, law.scale), lower, upper)
        )


class ResidualKernel(ValueModel):
    """An immutable native factory for one explicitly selected residual event.

    The constructed law treats its whole row vector as one correlated series
    and reuses `ResidualResponse` through an event reinterpretation.

    Parameters
    ----------
    kind : str
        One of arma, cosy, fixed, unstructured, sar_error, or sar_lag.
    matrix : tuple[tuple[float, ...], ...]
        Fixed covariance for fixed, or spatial weights for SAR. Known matrices
        are copied once, so posterior calls retain no caller-owned mutable data.
    stationary : bool
        Use stationary ARMA initialization rather than zero initial innovations.
    student : bool
        Use Student innovations and the natural ``nu``.
    """

    kind: Literal["arma", "cosy", "fixed", "unstructured", "sar_error", "sar_lag"]
    matrix: tuple[tuple[float, ...], ...] = dx.field(
        default=(), opaque=True, extras=opaque_by_value(_matrix_tuple)
    )
    stationary: bool = False
    student: bool = False

    @dx.model_validator()
    def _validate(self) -> None:
        if type(self.stationary) is not bool or type(self.student) is not bool:
            raise TypeError(
                "residual initialization and Student policy must be Boolean"
            )
        matrix = self.matrix
        if self.kind in {"fixed", "sar_error", "sar_lag"}:
            tensor = torch.tensor(matrix, dtype=torch.float64)
            if (
                tensor.ndim != 2
                or tensor.shape[0] != tensor.shape[1]
                or not tensor.shape[0]
            ):
                raise ValueError(
                    "fixed residual operators require a nonempty square matrix"
                )
            if self.kind == "fixed" and not bool(
                torch.allclose(tensor, tensor.T, atol=_tolerance(tensor), rtol=0)
            ):
                raise ValueError("fixed residual covariance must be symmetric")
        elif matrix:
            raise ValueError("this residual operator does not accept a fixed matrix")

    def layout(self, rows: int) -> ResidualLayout:
        """Describe one series over ``rows`` rows, prepared once per extent.

        A fixed covariance is factored and SAR eigenvalues are computed on
        the first request for a row count and reused by every later law.

        Parameters
        ----------
        rows : int
            Number of rows in the event.

        Returns
        -------
        ResidualLayout
            Single-series layout, with any known matrix prepared once.

        Raises
        ------
        ValueError
            If a known matrix does not match the rows.
        """
        return _kernel_layout(self, rows)

    def _prepare(self, rows: int) -> ResidualLayout:
        """Build the single-series layout for one row count.

        Parameters
        ----------
        rows : int
            Number of rows in the event.

        Returns
        -------
        ResidualLayout
            Layout with any known matrix factored or decomposed.

        Raises
        ------
        ValueError
            If a known matrix does not match the rows.
        """
        common = {
            "rows": rows,
            "order": tuple(range(rows)),
            "starts": (0,),
            "lengths": (rows,),
            "student": self.student,
        }
        if self.kind in {"fixed", "sar_error", "sar_lag"} and len(self.matrix) != rows:
            raise ValueError("fixed residual matrix and response rows disagree")
        if self.kind == "fixed":
            matrix = np.asarray(self.matrix, dtype=np.float64)
            return ResidualLayout(
                kind="fcor",
                matrix=matrix,
                matrix_factor=np.linalg.cholesky(matrix),
                **common,
            )
        if self.kind in {"sar_error", "sar_lag"}:
            weights, eigenvalues = sar_weights(np.asarray(self.matrix))
            return ResidualLayout(
                kind="sar_lag" if self.kind == "sar_lag" else "sar_error",
                weights=weights,
                eigenvalues=eigenvalues,
                **common,
            )
        if self.kind == "unstructured":
            return ResidualLayout(
                kind="unstr",
                time_index=tuple(range(rows)),
                times=rows,
                **common,
            )
        return ResidualLayout(kind=self.kind, stationary=self.stationary, **common)

    def __call__(self, parameters: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct the native law while retaining all parameter gradients.

        Parameters
        ----------
        parameters : Mapping[str, torch.Tensor]
            Natural-scale parameters: ``loc`` and ``scale`` always, with ``ar``
            and ``ma``, ``rho``, ``factor``, or ``nu`` as the operator requires.

        Returns
        -------
        Distribution
            The residual law over one event of rows; the conditional ARMA
            uses the innovation filter and every other operator the
            structured `ResidualResponse`.

        Raises
        ------
        ValueError
            If the location and scales disagree on their row axis, a scale is
            nonpositive, or an operator matrix does not match the rows.
        KeyError
            If a parameter the operator requires is absent.
        """
        loc, scale = parameters["loc"], parameters["scale"]
        _vector(loc, "residual location")
        _vector(scale, "residual scale")
        if loc.shape[-1] != scale.shape[-1]:
            raise ValueError(
                "residual location and positive scales must share their row axis"
            )
        demand(
            scale > 0,
            lambda: ValueError(
                "residual location and positive scales must share their row axis"
            ),
        )
        nu = parameters["nu"] if self.student else None
        if self.kind == "arma" and not self.stationary:
            return _ARMAInnovations(loc, scale, parameters["ar"], parameters["ma"], nu)
        if self.kind == "arma":
            factor = arma_factor(
                scale, parameters["ar"], parameters["ma"], stationary=True
            )
            if nu is not None:
                return MultivariateStudentT(nu, loc, scale_tril=factor)
            return MultivariateNormal(loc, scale_tril=factor)
        coefficients: dict[str, torch.Tensor] = {}
        if self.kind == "cosy":
            coefficients["cosy"] = parameters["rho"]
        elif self.kind in {"sar_error", "sar_lag"}:
            name = "lagsar" if self.kind == "sar_lag" else "errorsar"
            coefficients[name] = parameters["rho"]
        law = ResidualResponse(
            self.layout(loc.shape[-1]),
            loc,
            scale,
            coefficients,
            nu,
            parameters["factor"] if self.kind == "unstructured" else None,
        )
        return Independent(law, 1)


@functools.lru_cache(maxsize=256)
def _kernel_layout(kernel: ResidualKernel, rows: int) -> ResidualLayout:
    """Prepare and retain one kernel's layout for a row count.

    Parameters
    ----------
    kernel : ResidualKernel
        Immutable residual operator, hashed by value.
    rows : int
        Number of rows in the event.

    Returns
    -------
    ResidualLayout
        The prepared layout.
    """
    return kernel._prepare(rows)


def residual_catalog(
    kernel: ResidualKernel,
    *,
    name: str = "ResidualNormal",
    base: FamilyCatalog | None = None,
) -> FamilyCatalog:
    """Bind a residual operator through the ordinary checked family boundary.

    The event is one explicit correlated row vector. Callers retain grouping
    and observation identities. This contract never infers independence from
    an equal axis extent.

    Parameters
    ----------
    kernel : ResidualKernel
        The residual operator that constructs the native law.
    name : str
        Family name under which the operator is registered.
    base : FamilyCatalog or None
        Catalog to extend, or None for the default catalog.

    Returns
    -------
    FamilyCatalog
        The extended catalog, with one family whose parameters follow the
        operator kind and Student policy.
    """
    parameters = [
        FamilyParameter(
            name="loc",
            constraint="real",
            rank=1,
            event_dims=(0,),
            scalar_broadcast=False,
        ),
        FamilyParameter(
            name="scale",
            constraint="positive",
            rank=1,
            event_dims=(0,),
            scalar_broadcast=False,
        ),
    ]
    if kernel.kind == "arma":
        parameters.extend(
            FamilyParameter(name=key, constraint="real", rank=1, scalar_broadcast=False)
            for key in ("ar", "ma")
        )
    elif kernel.kind in {"cosy", "sar_error", "sar_lag"}:
        parameters.append(FamilyParameter(name="rho", constraint="real", rank=0))
    elif kernel.kind == "unstructured":
        parameters.append(
            FamilyParameter(
                name="factor",
                constraint="lower_cholesky",
                rank=2,
                event_dims=(0, 0),
                scalar_broadcast=False,
            )
        )
    if kernel.student:
        parameters.append(FamilyParameter(name="nu", constraint="positive", rank=0))
    record = DistributionFamily(
        name=name,
        parameters=tuple(parameters),
        support="real",
        element=REAL,
        event_rank=1,
        event_sources=("loc",),
        discrete=False,
        reparameterizable=kernel.kind not in {"sar_error", "sar_lag"},
        compositional=False,
    )
    return (FamilyCatalog() if base is None else base).with_family(record, kernel)


__all__: list[str] = []
