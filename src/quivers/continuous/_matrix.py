"""Matrix-valued distribution families PyTorch does not ship.

`InverseWishart` is the law of the inverse of a Wishart draw, and
`MatrixNormal` the normal law on real matrices whose covariance is the
Kronecker product of a row and a column covariance. Both sample with a
reparameterization and score in closed form, so the compiled runtime, the
conditional morphisms, and the transpile layer's shape inference share one
implementation of each.

`PositiveDefiniteTransform` maps unconstrained coordinates to a
positive-definite matrix through its Cholesky factor, with the log
absolute Jacobian determinant inference needs to move a matrix-valued
site to and from an unconstrained space.
"""

from __future__ import annotations

import math

import torch
import torch.distributions as D
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transforms import Transform


class InverseWishart(D.Distribution):
    """The inverse of a Wishart draw.

     With ``W ~ Wishart(df, scale_tril=L)``, the matrix ``Sigma = W^-1``
     has density ``p_W(Sigma^-1) |det Sigma|^-(d + 1)``, the second factor
     the Jacobian of matrix inversion on ``d x d`` symmetric matrices
    . In the conventional parameterization
     ``IW(df, Psi)`` the scale is ``Psi = (L L^T)^-1``.

     Parameters
     ----------
     df : torch.Tensor
         Degrees of freedom, greater than ``d - 1``.
     scale_tril : torch.Tensor
         The lower Cholesky factor of the scale of the Wishart whose
         inverse is drawn, ``(..., d, d)``.
     validate_args : bool | None
         Whether to validate the arguments and values.
    """

    arg_constraints: dict[str, constraints.Constraint] = {
        "df": constraints.positive,
        "scale_tril": constraints.lower_cholesky,
    }
    support: constraints.Constraint = constraints.positive_definite
    has_rsample = True

    def __init__(
        self,
        df: Tensor,
        scale_tril: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self._wishart = D.Wishart(
            df=df, scale_tril=scale_tril, validate_args=validate_args
        )
        self.df = self._wishart.df
        self.scale_tril = self._wishart.scale_tril
        super().__init__(
            batch_shape=self._wishart.batch_shape,
            event_shape=self._wishart.event_shape,
            validate_args=validate_args,
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw by inverting a Wishart draw.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample axes.

        Returns
        -------
        torch.Tensor
            The draws, ``(*sample_shape, *batch_shape, d, d)``.
        """
        draw = self._wishart.rsample(torch.Size(sample_shape))
        return torch.cholesky_inverse(torch.linalg.cholesky(draw))

    def log_prob(self, value: Tensor) -> Tensor:
        """The log density at a positive-definite matrix.

        Parameters
        ----------
        value : torch.Tensor
            The matrices, ``(..., d, d)``.

        Returns
        -------
        torch.Tensor
            The log densities.
        """
        if self._validate_args:
            self._validate_sample(value)
        factor = torch.linalg.cholesky(value)
        inverse = torch.cholesky_inverse(factor)
        size = value.shape[-1]
        log_det = 2.0 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return self._wishart.log_prob(inverse) - (size + 1) * log_det


class MatrixNormal(D.Distribution):
    """The normal law on ``n x p`` matrices with Kronecker covariance.

    ``X ~ MN(M, U, V)`` when its row-major vectorization is normal with
    mean that of ``M`` and covariance ``U (x) V``: entries ``X[i, j]`` and
    ``X[k, l]`` covary as ``U[i, k] V[j, l]``. With
    ``U = A A^T`` and ``V = B B^T`` a draw is ``M + A Z B^T`` for a
    standard normal ``Z``, and the log density is
    ``-(np/2) log 2 pi - (p/2) log det U - (n/2) log det V
    - tr(V^-1 R^T U^-1 R) / 2`` at a residual ``R = X - M``.

    Parameters
    ----------
    loc : torch.Tensor
        The mean, ``(..., n, p)``.
    row_covariance : torch.Tensor
        The row covariance ``U``, ``(..., n, n)``.
    col_covariance : torch.Tensor
        The column covariance ``V``, ``(..., p, p)``.
    validate_args : bool | None
        Whether to validate the arguments and values.
    """

    arg_constraints: dict[str, constraints.Constraint] = {
        "loc": constraints.independent(constraints.real, 2),
        "row_covariance": constraints.positive_definite,
        "col_covariance": constraints.positive_definite,
    }
    support: constraints.Constraint = constraints.independent(constraints.real, 2)
    has_rsample = True

    def __init__(
        self,
        loc: Tensor,
        row_covariance: Tensor,
        col_covariance: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = loc
        self.row_covariance = row_covariance
        self.col_covariance = col_covariance
        self._row_tril = torch.linalg.cholesky(row_covariance)
        self._col_tril = torch.linalg.cholesky(col_covariance)
        batch_shape = torch.broadcast_shapes(
            loc.shape[:-2], row_covariance.shape[:-2], col_covariance.shape[:-2]
        )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=loc.shape[-2:],
            validate_args=validate_args,
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw ``M + A Z B^T``.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample axes.

        Returns
        -------
        torch.Tensor
            The draws, ``(*sample_shape, *batch_shape, n, p)``.
        """
        shape = self._extended_shape(torch.Size(sample_shape))
        noise = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self._row_tril @ noise @ self._col_tril.transpose(-1, -2)

    def log_prob(self, value: Tensor) -> Tensor:
        """The log density at a matrix.

        Parameters
        ----------
        value : torch.Tensor
            The matrices, ``(..., n, p)``.

        Returns
        -------
        torch.Tensor
            The log densities.
        """
        if self._validate_args:
            self._validate_sample(value)
        rows, cols = self.event_shape
        residual = value - self.loc
        whitened = torch.linalg.solve_triangular(
            self._row_tril.expand(*residual.shape[:-2], rows, rows),
            residual,
            upper=False,
        )
        whitened = torch.linalg.solve_triangular(
            self._col_tril.expand(*residual.shape[:-2], cols, cols),
            whitened.transpose(-1, -2),
            upper=False,
        )
        quadratic = whitened.pow(2).sum((-1, -2))
        row_log_det = 2.0 * self._row_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        col_log_det = 2.0 * self._col_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (
            rows * cols * math.log(2.0 * math.pi)
            + cols * row_log_det
            + rows * col_log_det
            + quadratic
        )


class PositiveDefiniteTransform(Transform):
    """Unconstrained coordinates to a positive-definite matrix.

    The ``d (d + 1) / 2`` coordinates fill a lower-triangular matrix in
    row-major order, its diagonal through the exponential, which is the
    Cholesky factor ``L`` of the image ``L L^T``. The log absolute
    Jacobian determinant is that of the exponential on the diagonal,
    ``sum_i x_ii``, plus that of ``L -> L L^T`` on lower-triangular
    matrices with a positive diagonal, ``d log 2 + sum_i (d - i + 1)
    log L_ii`` for ``i = 1, ..., d``: in total
    ``d log 2 + sum_i (d - i + 2) x_ii``.

    Parameters
    ----------
    dim : int
        The matrix size ``d``.
    cache_size : int
        The size of the transform's cache, as `Transform` takes it.
    """

    domain = constraints.independent(constraints.real, 1)
    codomain = constraints.positive_definite
    bijective = True

    def __init__(self, dim: int, cache_size: int = 0) -> None:
        """Build the transform for ``dim x dim`` matrices.

        Parameters
        ----------
        dim : int
            The matrix size.
        cache_size : int
            The size of the transform's cache, as `Transform` takes it.
        """
        super().__init__(cache_size=cache_size)
        self.dim = int(dim)
        rows, cols = torch.tril_indices(self.dim, self.dim)
        self._rows = rows
        self._cols = cols
        self._diagonal = (rows == cols).nonzero().squeeze(-1)
        # Row i of the factor (one-based) contributes d - i + 2 times its
        # diagonal coordinate: d - i + 1 from the product, one from the
        # exponential.
        self._weights = torch.tensor(
            [self.dim - index + 1.0 for index in range(self.dim)]
        )

    def __eq__(self, other: Transform) -> bool:  # type: ignore[override]
        """Whether another transform is this one.

        Parameters
        ----------
        other : Transform
            The other transform.

        Returns
        -------
        bool
            Whether it maps to positive-definite matrices of the same size.
        """
        return isinstance(other, PositiveDefiniteTransform) and other.dim == self.dim

    def __hash__(self) -> int:
        """A hash agreeing with equality.

        Returns
        -------
        int
            The hash of the class and the matrix size.
        """
        return hash((PositiveDefiniteTransform, self.dim))

    def _factor(self, x: Tensor) -> Tensor:
        """The Cholesky factor the coordinates fill.

        Parameters
        ----------
        x : torch.Tensor
            The coordinates, ``(..., d (d + 1) / 2)``.

        Returns
        -------
        torch.Tensor
            The factor, ``(..., d, d)``.
        """
        entries = x.clone()
        entries[..., self._diagonal] = x[..., self._diagonal].exp()
        factor = x.new_zeros(*x.shape[:-1], self.dim, self.dim)
        factor[..., self._rows, self._cols] = entries
        return factor

    def _call(self, x: Tensor) -> Tensor:
        """Map coordinates to the matrix ``L L^T``.

        Parameters
        ----------
        x : torch.Tensor
            The coordinates.

        Returns
        -------
        torch.Tensor
            The positive-definite matrix.
        """
        factor = self._factor(x)
        return factor @ factor.transpose(-1, -2)

    def _inverse(self, y: Tensor) -> Tensor:
        """Map a positive-definite matrix to its coordinates.

        Parameters
        ----------
        y : torch.Tensor
            The matrix.

        Returns
        -------
        torch.Tensor
            Its Cholesky factor's lower triangle, the diagonal logged.
        """
        factor = torch.linalg.cholesky(y)
        entries = factor[..., self._rows, self._cols]
        logged = entries.clone()
        logged[..., self._diagonal] = entries[..., self._diagonal].log()
        return logged

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """The log absolute Jacobian determinant of the map at ``x``.

        Parameters
        ----------
        x : torch.Tensor
            The coordinates.
        y : torch.Tensor
            Their image, which the determinant does not need.

        Returns
        -------
        torch.Tensor
            ``d log 2 + sum_i (d - i + 2) x_ii``.
        """
        del y
        diagonal = x[..., self._diagonal]
        weights = self._weights.to(dtype=x.dtype, device=x.device)
        return self.dim * math.log(2.0) + (weights * diagonal).sum(-1)

    def forward_shape(self, shape: torch.Size) -> torch.Size:
        """The shape of the image of coordinates of a shape.

        Parameters
        ----------
        shape : torch.Size
            The coordinates' shape.

        Returns
        -------
        torch.Size
            The matrices' shape.
        """
        return torch.Size((*shape[:-1], self.dim, self.dim))

    def inverse_shape(self, shape: torch.Size) -> torch.Size:
        """The shape of the coordinates of matrices of a shape.

        Parameters
        ----------
        shape : torch.Size
            The matrices' shape.

        Returns
        -------
        torch.Size
            The coordinates' shape.
        """
        return torch.Size((*shape[:-2], self.dim * (self.dim + 1) // 2))
