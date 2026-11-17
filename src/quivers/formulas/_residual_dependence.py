"""Residual operators construct existing Gaussian and Student laws.

Each invocation describes one correlated predictive unit. Independent temporal
or spatial groups use separate invocations, including separate Student radial
variables. The operators retain gradients and contain no likelihood scorer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.distributions import (
    Distribution,
    Independent,
    MultivariateNormal,
    Normal,
    StudentT,
    TransformedDistribution,
    Transform,
    constraints,
)

from quivers.continuous._multivariate_student_t import MultivariateStudentT
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._group_covariance import _matrix_values, _tolerance
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL


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
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} requires finite floating values")


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
    if not bool((scale > 0).all()):
        raise ValueError("residual scales must be positive")
    for name, value in (("ar", ar), ("ma", ma)):
        if (
            value.ndim < 1
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"{name} requires a finite coefficient vector")
    count = scale.shape[-1]
    batch = torch.broadcast_shapes(scale.shape[:-1], ar.shape[:-1], ma.shape[:-1])
    identity = torch.eye(count, dtype=scale.dtype, device=scale.device).expand(
        *batch, count, count
    )
    if stationary:
        if ar.shape[-1] > 1 or ma.shape[-1] > 1:
            raise ValueError("stationary ARMA covariance requires orders at most one")
        phi = ar[..., 0] if ar.shape[-1] else scale.new_zeros(batch)
        theta = ma[..., 0] if ma.shape[-1] else scale.new_zeros(batch)
        if not bool((phi.abs() < 1).all()):
            raise ValueError("stationary ARMA requires abs(ar) < 1")
        positions = torch.arange(count, device=scale.device)
        lag = (positions[:, None] - positions[None, :]).abs()
        variance = (1 + theta.square() + 2 * phi * theta) / (1 - phi.square())
        first = (1 + phi * theta) * (phi + theta) / (1 - phi.square())
        powers = [torch.ones_like(phi)]
        for _ in range(max(0, count - 2)):
            powers.append(powers[-1] * phi)
        decay = torch.stack(powers, dim=-1)[..., (lag - 1).clamp_min(0)]
        covariance = first[..., None, None] * decay
        covariance = torch.where(lag == 0, variance[..., None, None], covariance)
        correlation_factor = torch.linalg.cholesky(covariance)
        return scale[..., :, None] * correlation_factor
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
    if not bool(torch.isfinite(rho).all()) or not bool((scale > 0).all()):
        raise ValueError("residual scales and correlation must be finite and valid")
    if count > 1 and not bool(((rho > -1 / (count - 1)) & (rho < 1)).all()):
        raise ValueError(
            "compound symmetry correlation is outside its positive-definite domain"
        )
    identity = torch.eye(count, dtype=scale.dtype, device=scale.device)
    covariance = identity + rho[..., None, None] * (1 - identity)
    return scale[..., :, None] * torch.linalg.cholesky(covariance)


def sar_parameters(
    loc: torch.Tensor,
    scale: torch.Tensor,
    rho: torch.Tensor,
    weights: torch.Tensor,
    *,
    lag: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform an existing residual law by a simultaneous autoregression.

    Error SAR retains ``loc``. Lag SAR solves the same spatial operator for
    the location. Both return the covariance of transformed independent
    innovations, allowing the existing Gaussian/Student constructors to own
    normalization, scoring, and sampling.

    Parameters
    ----------
    loc : torch.Tensor
        Residual location, with one trailing coordinate per response row.
    scale : torch.Tensor
        Positive innovation scales on the same rows.
    rho : torch.Tensor
        Spatial autoregression coefficient, with optional leading batches.
    weights : torch.Tensor
        Finite square spatial weight matrix over the response rows.
    lag : bool
        Solve the spatial operator for the location as well (lag SAR). False
        retains ``loc`` (error SAR).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The transformed location and the covariance of the transformed
        independent innovations.

    Raises
    ------
    ValueError
        If the weights are not a finite square matrix over the rows, the
        correlation or scales are invalid, or the spatial operator is singular.
    """
    _vector(loc, "residual location")
    _vector(scale, "residual scale")
    count = loc.shape[-1]
    if weights.shape[-2:] != (count, count) or not bool(torch.isfinite(weights).all()):
        raise ValueError(
            "SAR weights must be a finite square matrix over response rows"
        )
    if not bool(torch.isfinite(rho).all()) or not bool((scale > 0).all()):
        raise ValueError(
            "SAR correlation and innovation scales must be finite and valid"
        )
    operator = (
        torch.eye(count, dtype=loc.dtype, device=loc.device)
        - rho[..., None, None] * weights
    )
    innovations, info = torch.linalg.solve_ex(operator, torch.diag_embed(scale))
    if bool((info != 0).any()) or not bool(torch.isfinite(innovations).all()):
        raise ValueError("SAR operator must be nonsingular")
    mean = torch.linalg.solve(operator, loc.unsqueeze(-1)).squeeze(-1) if lag else loc
    return mean, innovations @ innovations.mT


class _ARMAFilter(Transform):
    """Invert a finite innovation filter in linear time for fixed lag orders.

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


@dataclass(frozen=True)
class ResidualKernel:
    """An immutable native factory for one explicitly selected residual block.

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
        Use the existing multivariate Student law and its natural ``nu``.
    """

    kind: Literal["arma", "cosy", "fixed", "unstructured", "sar_error", "sar_lag"]
    matrix: tuple[tuple[float, ...], ...] = ()
    stationary: bool = False
    student: bool = False
    _fixed_factor: tuple[tuple[float, ...], ...] = field(
        default=(), init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate the operator and copy any known matrix once.

        Raises
        ------
        ValueError
            If the operator is unknown, a fixed matrix is missing, malformed,
            or asymmetric, or a matrix is given to an operator without one.
        TypeError
            If the initialization or Student policy is not Boolean.
        """
        if self.kind not in {
            "arma",
            "cosy",
            "fixed",
            "unstructured",
            "sar_error",
            "sar_lag",
        }:
            raise ValueError("unknown residual covariance operator")
        if type(self.stationary) is not bool or type(self.student) is not bool:
            raise TypeError(
                "residual initialization and Student policy must be Boolean"
            )
        matrix = _matrix_values(self.matrix)
        object.__setattr__(self, "matrix", matrix)
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
            if self.kind == "fixed":
                if not bool(
                    torch.allclose(tensor, tensor.T, atol=_tolerance(tensor), rtol=0)
                ):
                    raise ValueError("fixed residual covariance must be symmetric")
                object.__setattr__(
                    self, "_fixed_factor", _matrix_values(torch.linalg.cholesky(tensor))
                )
        elif matrix:
            raise ValueError("this residual operator does not accept a fixed matrix")

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
            A multivariate Gaussian or Student law, or an ARMA innovation law
            under conditional initialization.

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
        if loc.shape[-1] != scale.shape[-1] or not bool((scale > 0).all()):
            raise ValueError(
                "residual location and positive scales must share their row axis"
            )
        if self.kind == "arma" and not self.stationary:
            return _ARMAInnovations(
                loc,
                scale,
                parameters["ar"],
                parameters["ma"],
                parameters["nu"] if self.student else None,
            )
        if self.kind == "arma":
            factor = arma_factor(
                scale, parameters["ar"], parameters["ma"], stationary=self.stationary
            )
        elif self.kind == "cosy":
            factor = compound_symmetry_factor(scale, parameters["rho"])
        elif self.kind == "unstructured":
            correlation = parameters["factor"]
            if correlation.shape[-2:] != (loc.shape[-1], loc.shape[-1]):
                raise ValueError(
                    "residual correlation factor must retain response-row coordinates"
                )
            factor = scale[..., :, None] * correlation
        elif self.kind == "fixed":
            fixed = loc.new_tensor(self.matrix)
            if fixed.shape != (loc.shape[-1], loc.shape[-1]):
                raise ValueError("fixed residual covariance and response rows disagree")
            factor = scale[..., :, None] * loc.new_tensor(self._fixed_factor)
        else:
            loc, covariance = sar_parameters(
                loc,
                scale,
                parameters["rho"],
                loc.new_tensor(self.matrix),
                lag=self.kind == "sar_lag",
            )
            factor = torch.linalg.cholesky(covariance)
        if self.student:
            return MultivariateStudentT(parameters["nu"], loc, scale_tril=factor)
        return MultivariateNormal(loc, scale_tril=factor)


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
        reparameterizable=True,
        compositional=False,
    )
    return (FamilyCatalog() if base is None else base).with_family(record, kernel)
