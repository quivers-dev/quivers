"""Elliptical Student responses using the existing Gaussian factor geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.distributions import Chi2, Distribution, MultivariateNormal, constraints
from torch.distributions.multivariate_normal import _batch_mahalanobis

from quivers.continuous._closed_probability import _log_gamma_increment


class MultivariateStudentT(Distribution):
    """A multivariate Student law with one shared radial scale per vector.

    Parameters
    ----------
    df : Tensor or float
        Positive degrees of freedom, broadcast over Gaussian batch axes.
    loc : Tensor
        Location with trailing event dimension ``D``.
    covariance_matrix, precision_matrix, scale_tril : Tensor or None
        Exactly one Gaussian scale representation. The scale matrix is not
        the response covariance; the latter is ``df/(df-2)`` times the scale
        when ``df > 2``. Factors follow ``MultivariateNormal`` conventions.
    validate_args : bool or None
        Torch argument and observation validation policy.

    Notes
    -----
    A single chi-squared variate scales every coordinate in each vector.
    Independent univariate Student distributions have a different joint law.
    The location is the response mean only for degrees of freedom above one.
    """

    has_rsample = True
    _atomless = True
    _batch_parameter_ranks = {"df": 0, "loc": 1, "scale_tril": 2}

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Read the checked heads without materializing covariance properties.

        Returns
        -------
        dict[str, Constraint]
            Actual stored Gaussian factor and Student head constraints.
        """
        return {
            "df": constraints.positive,
            "loc": constraints.real_vector,
            "scale_tril": constraints.lower_cholesky,
        }

    @property
    def support(self) -> constraints.Constraint:
        """Return the real-vector observation carrier.

        Returns
        -------
        Constraint
            One final unconstrained real event axis.
        """
        return constraints.real_vector

    def __init__(
        self,
        df: Tensor | float,
        loc: Tensor,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        df = torch.as_tensor(
            df,
            device=loc.device,
            dtype=None if isinstance(df, Tensor) else loc.dtype,
        )
        dtype = torch.promote_types(df.dtype, loc.dtype)
        for value in (covariance_matrix, precision_matrix, scale_tril):
            if value is not None:
                dtype = torch.promote_types(dtype, value.dtype)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        normal = MultivariateNormal(
            loc.to(dtype),
            covariance_matrix=None
            if covariance_matrix is None
            else covariance_matrix.to(dtype),
            precision_matrix=None
            if precision_matrix is None
            else precision_matrix.to(dtype),
            scale_tril=None if scale_tril is None else scale_tril.to(dtype),
            validate_args=validate_args,
        )
        batch = torch.broadcast_shapes(df.shape, normal.batch_shape)
        self._normal = normal.expand(batch)
        self.df = df.to(dtype).expand(batch)
        self.loc = self._normal.loc
        self.scale_tril = self._normal.scale_tril
        super().__init__(batch, normal.event_shape, validate_args=validate_args)

    @property
    def covariance_matrix(self) -> Tensor:
        """Return the Gaussian scale matrix, following Torch parameter names.

        Returns
        -------
        Tensor
            Scale matrices with trailing shape ``(D,D)``.
        """
        return self._normal.covariance_matrix

    @property
    def precision_matrix(self) -> Tensor:
        """Return the inverse Gaussian scale matrix.

        Returns
        -------
        Tensor
            Inverse scale matrices with trailing shape ``(D,D)``.
        """
        return self._normal.precision_matrix

    @property
    def mean(self) -> Tensor:
        """Return the finite response mean where it exists.

        Returns
        -------
        Tensor
            Location for ``df > 1`` and NaN otherwise.
        """
        return torch.where(self.df[..., None] > 1, self.loc, torch.nan)

    @property
    def variance(self) -> Tensor:
        """Return marginal variances with Student moment existence boundaries.

        Returns
        -------
        Tensor
            Finite variances above two degrees of freedom, infinity between
            one and two, and NaN at or below one.
        """
        finite = self.df > 2
        df = torch.where(finite, self.df, 3.0)
        variance = self._normal.variance * (df / (df - 2))[..., None]
        return torch.where(
            finite[..., None],
            variance,
            torch.where(self.df[..., None] > 1, torch.inf, torch.nan),
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluate the marginal vector density without auxiliary latent sites.

        Parameters
        ----------
        value : Tensor
            Observations ending in the declared vector event axis.

        Returns
        -------
        Tensor
            One log density per sample and batch coordinate.
        """
        if self._validate_args:
            self._validate_sample(value)
        width = self.event_shape[0]
        factor = self._normal._unbroadcasted_scale_tril
        squared = _batch_mahalanobis(factor, value - self.loc)
        log_det = factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        half = self.df / 2
        increment = torch.full_like(half, width / 2)
        return (
            _log_gamma_increment(half, increment)
            - (width / 2) * (self.df.log() + math.log(math.pi))
            - log_det
            - ((self.df + width) / 2) * torch.log1p(squared / self.df)
        )

    def rsample(self, sample_shape: Sequence[int] = torch.Size()) -> Tensor:
        """Draw pathwise using one radial chi-squared scale per vector.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample axes.

        Returns
        -------
        Tensor
            Draws with sample, batch, and final vector event dimensions.
        """
        normal = self._normal.rsample(torch.Size(sample_shape))
        radial = Chi2(self.df).rsample(torch.Size(sample_shape))
        return self.loc + (normal - self.loc) * torch.sqrt(self.df / radial)[..., None]

    def expand(
        self, batch_shape: Sequence[int], _instance=None
    ) -> MultivariateStudentT:
        """Expand only batch axes while retaining the shared factor.

        Parameters
        ----------
        batch_shape : torch.Size
            Broadcast-compatible new batch dimensions.
        _instance : MultivariateStudentT or None
            Optional uninitialized expansion instance, following Torch.

        Returns
        -------
        MultivariateStudentT
            Expanded distribution without another matrix factorization.
        """
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        new._normal = self._normal.expand(batch_shape)
        new.df = self.df.expand(batch_shape)
        new.loc = new._normal.loc
        new.scale_tril = new._normal.scale_tril
        super(MultivariateStudentT, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


__all__: list[str] = []
