"""Labeled covariance and support coordinates for Gaussian grouping levels."""

from __future__ import annotations

from collections.abc import Sequence

import didactic.api as dx
import torch

from quivers._values import ValueModel, opaque_by_value


def _matrix_values(value):
    """Freeze a fixed real matrix without retaining mutable input storage.

    Parameters
    ----------
    value : array-like
        Known covariance or coordinate matrix.

    Returns
    -------
    tuple[tuple[float, ...], ...]
        Immutable Double-precision values.

    Raises
    ------
    ValueError
        If the input is not a finite fixed real matrix.
    """
    tensor = (
        value
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value, dtype=torch.float64)
    )
    if tensor.numel() == 0 and tensor.ndim == 1:
        tensor = tensor.reshape(0, 0)
    if tensor.ndim != 2 or tensor.is_complex() or tensor.dtype == torch.bool:
        raise ValueError("group covariance must be a real matrix")
    if tensor.requires_grad:
        raise ValueError(
            "group covariance is known data and must not require gradients"
        )
    tensor = tensor.to(dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("group covariance entries must be finite")
    return tuple(tuple(row) for row in tensor.tolist())


class GroupCovariance(ValueModel):
    """A known covariance between explicitly labeled grouping levels.

    Parameters
    ----------
    levels : tuple[str, ...]
        Unique authored labels in matrix row and column order. The matrix can
        include future levels absent from the fitting data.
    matrix : tuple[tuple[float, ...], ...]
        Finite symmetric covariance, copied from nested numeric sequences or
        a fixed Tensor. Its scale is retained; it need not be a correlation.
    rank : int or None
        None requires positive definiteness. An explicit nonnegative rank
        declares a positive-semidefinite law on its actual affine support.
        Eigenvalues below the declared rank must vanish within a relative
        Double-precision eigensolver tolerance; negative eigenvalues beyond
        that tolerance are rejected.
    """

    levels: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...] | torch.Tensor = dx.field(
        opaque=True, extras=opaque_by_value(_matrix_values)
    )
    rank: int | None = None

    @dx.model_validator()
    def _validate(self) -> None:
        """Check label alignment and the declared covariance support.

        Raises
        ------
        ValueError
            If labels, symmetry, definiteness, or the declared rank disagree.
        """
        if not self.levels or any(not isinstance(x, str) or not x for x in self.levels):
            raise ValueError("group covariance labels must be nonempty strings")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("group covariance labels must be unique")
        matrix = self.tensor()
        count = len(self.levels)
        if matrix.shape != (count, count):
            raise ValueError("group covariance must have one row and column per label")
        tolerance = _tolerance(matrix)
        if not bool(torch.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance)):
            raise ValueError("group covariance must be symmetric")
        matrix = matrix * 0.5 + matrix.T * 0.5
        if self.rank is None:
            _, info = torch.linalg.cholesky_ex(matrix)
            if int(info) != 0:
                raise ValueError(
                    "group covariance must be positive definite; declare rank for PSD support"
                )
        else:
            if type(self.rank) is not int or not 0 <= self.rank <= count:
                raise ValueError(
                    "group covariance rank must be an integer between zero and its size"
                )
            eigenvalues = torch.linalg.eigvalsh(matrix)
            rank = self.rank
            if bool((eigenvalues < -tolerance).any()):
                raise ValueError("group covariance must be positive semidefinite")
            if rank and float(eigenvalues[-rank]) <= tolerance:
                raise ValueError(
                    "declared group covariance rank exceeds its positive spectrum"
                )
            if count > rank and bool(
                (eigenvalues[: count - rank].abs() > tolerance).any()
            ):
                raise ValueError(
                    "declared group covariance rank omits nonzero eigenvalues"
                )

    def tensor(self) -> torch.Tensor:
        """Read a fresh Double-precision covariance matrix.

        Returns
        -------
        torch.Tensor
            Independent CPU storage; modifying it cannot change this value.
        """
        return torch.tensor(_matrix_values(self.matrix), dtype=torch.float64)

    def subset(self, levels: Sequence[str]) -> GroupCovariance:
        """Retain a labeled principal submatrix in the requested order.

        Parameters
        ----------
        levels : Sequence[str]
            Unique known grouping labels.

        Returns
        -------
        GroupCovariance
            Reordered covariance, retaining explicit PSD support when declared.

        Raises
        ------
        ValueError
            If a requested label is unknown or repeated.
        """
        labels = tuple(levels)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError(
                "group covariance subset labels must be nonempty and unique"
            )
        positions = {label: index for index, label in enumerate(self.levels)}
        missing = set(labels) - positions.keys()
        if missing:
            raise ValueError(
                f"group covariance has no entries for levels {sorted(missing)!r}"
            )
        indices = torch.tensor([positions[label] for label in labels])
        matrix = self.tensor()[indices[:, None], indices[None, :]]
        rank = None
        if self.rank is not None:
            rank = int((torch.linalg.eigvalsh(matrix) > _tolerance(matrix)).sum())
        return GroupCovariance(levels=labels, matrix=matrix, rank=rank)

    def basis(self, levels: Sequence[str]) -> GroupCovarianceBasis:
        """Factor a principal covariance without adding artificial dimensions.

        Parameters
        ----------
        levels : Sequence[str]
            Ordered labels used by one emitted Gaussian block.

        Returns
        -------
        GroupCovarianceBasis
            Orthogonal support projection and exact positive coordinate law.
        """
        covariance = self.subset(levels)
        matrix = covariance.tensor()
        if covariance.rank is None or covariance.rank == len(covariance.levels):
            projection = torch.eye(len(covariance.levels), dtype=torch.float64)
            coordinate = matrix
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            rank = covariance.rank
            projection = eigenvectors[:, -rank:] if rank else eigenvectors[:, :0]
            coordinate = torch.diag(eigenvalues[-rank:] if rank else eigenvalues[:0])
        cholesky = (
            torch.linalg.cholesky(coordinate) if coordinate.numel() else coordinate
        )
        return GroupCovarianceBasis(
            levels=covariance.levels,
            projection=_matrix_values(projection),
            covariance=_matrix_values(coordinate),
            coordinate_factor=_matrix_values(cholesky),
            sampling_factor=_matrix_values(projection @ cholesky),
        )

    def conditional(
        self, fitted_levels: Sequence[str], new_levels: Sequence[str]
    ) -> tuple[torch.Tensor, GroupCovarianceBasis]:
        """Condition new Gaussian levels on the complete fitted level vector.

        Parameters
        ----------
        fitted_levels, new_levels : Sequence[str]
            Disjoint known labels, in retained and requested order.

        Returns
        -------
        tuple[torch.Tensor, GroupCovarianceBasis]
            Mean map ``A_new,fitted A_fitted,fitted^+`` and a support factor
            of the Schur-complement conditional row covariance. Nonzero
            hypermeans are applied outside this deviation-space operation.

        Raises
        ------
        ValueError
            If labels overlap or the conditional covariance is not PSD.
        """
        fitted, new = tuple(fitted_levels), tuple(new_levels)
        if set(fitted) & set(new):
            raise ValueError(
                "conditional group covariance requires disjoint level sets"
            )
        joined = self.subset((*fitted, *new)).tensor()
        count = len(fitted)
        aa, ba, bb = (
            joined[:count, :count],
            joined[count:, :count],
            joined[count:, count:],
        )
        if self.rank is None:
            factor = torch.linalg.cholesky(aa)
            mean_map = torch.cholesky_solve(ba.T, factor).T
        else:
            eigenvalues, vectors = torch.linalg.eigh(aa)
            positive = eigenvalues > _tolerance(aa)
            inverse = (vectors[:, positive] / eigenvalues[positive]) @ vectors[
                :, positive
            ].T
            mean_map = ba @ inverse
        schur = bb - mean_map @ ba.T
        schur = (schur + schur.T) * 0.5
        values, vectors = torch.linalg.eigh(schur)
        tolerance = _tolerance(joined) * 8
        if bool((values < -tolerance).any()):
            raise ValueError(
                "conditional group covariance is not positive semidefinite"
            )
        if self.rank is None and bool((values <= 0).any()):
            raise ValueError(
                "conditional covariance lost positive definiteness at Double precision"
            )
        positive = values > (0 if self.rank is None else tolerance)
        return mean_map, GroupCovarianceBasis(
            levels=new,
            projection=_matrix_values(vectors[:, positive]),
            covariance=_matrix_values(torch.diag(values[positive])),
            coordinate_factor=_matrix_values(torch.diag(values[positive].sqrt())),
            sampling_factor=_matrix_values(
                vectors[:, positive] * values[positive].sqrt()
            ),
        )


class GroupCovarianceBasis(ValueModel):
    """Retain the physical-level map and a nonsingular support-coordinate law.

    Parameters
    ----------
    levels : tuple[str, ...]
        Physical grouping labels.
    projection : tuple[tuple[float, ...], ...]
        Orthonormal columns mapping support coordinates to physical levels.
    covariance : tuple[tuple[float, ...], ...]
        Positive-definite support covariance; an empty matrix denotes rank zero.
    coordinate_factor : tuple[tuple[float, ...], ...]
        Cached Cholesky factor of the support covariance.
    sampling_factor : tuple[tuple[float, ...], ...]
        Cached projection times coordinate_factor. Prediction materializes
        this factor without repeating an eigendecomposition or Cholesky.
    """

    levels: tuple[str, ...]
    projection: tuple[tuple[float, ...], ...] = dx.field(
        opaque=True, extras=opaque_by_value(_matrix_values)
    )
    covariance: tuple[tuple[float, ...], ...] = dx.field(
        opaque=True, extras=opaque_by_value(_matrix_values)
    )

    coordinate_factor: tuple[tuple[float, ...], ...] = dx.field(
        opaque=True, extras=opaque_by_value(_matrix_values)
    )

    sampling_factor: tuple[tuple[float, ...], ...] = dx.field(
        opaque=True, extras=opaque_by_value(_matrix_values)
    )

    @dx.model_validator()
    def _validate(self) -> None:
        """Check the retained support decomposition and cached factors.

        Raises
        ------
        ValueError
            If matrix shapes or factorization identities are inconsistent.
        """
        projection = torch.tensor(_matrix_values(self.projection), dtype=torch.float64)
        covariance = torch.tensor(
            _matrix_values(self.covariance), dtype=torch.float64
        ).reshape(self.rank, self.rank)
        coordinate = torch.tensor(
            _matrix_values(self.coordinate_factor), dtype=torch.float64
        ).reshape(self.rank, self.rank)
        sampling = torch.tensor(
            _matrix_values(self.sampling_factor), dtype=torch.float64
        )
        if (
            projection.shape != (len(self.levels), self.rank)
            or sampling.shape != projection.shape
        ):
            raise ValueError(
                "group covariance basis has inconsistent physical/support axes"
            )
        tolerance = 256 * torch.finfo(torch.float64).eps * max(1, len(self.levels))
        if not torch.allclose(
            projection.T @ projection,
            torch.eye(self.rank, dtype=torch.float64),
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("group covariance support projection must be orthonormal")
        if not torch.allclose(coordinate, torch.tril(coordinate), atol=0, rtol=0) or (
            self.rank and not bool((coordinate.diag() > 0).all())
        ):
            raise ValueError(
                "group covariance coordinate factor must be nonsingular lower triangular"
            )
        if not torch.allclose(
            coordinate @ coordinate.T,
            covariance,
            atol=8 * _tolerance(covariance),
            rtol=tolerance,
        ):
            raise ValueError(
                "group covariance coordinate factor disagrees with its covariance"
            )
        if not torch.allclose(
            projection @ coordinate,
            sampling,
            atol=8 * _tolerance(sampling),
            rtol=tolerance,
        ):
            raise ValueError(
                "group covariance sampling factor disagrees with its support projection"
            )

    @property
    def rank(self) -> int:
        """Return the number of stochastic level coordinates.

        Returns
        -------
        int
            Support dimension, including zero for a deterministic deviation.
        """
        return len(self.covariance)

    def tensors(
        self, *, like: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize projection, coordinate covariance, and sampling factor.

        Parameters
        ----------
        like : torch.Tensor or None
            Optional floating dtype and device context.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``Q``, ``C``, and ``Q chol(C)`` with explicit physical/support axes.
        """
        dtype = torch.float64 if like is None else like.dtype
        device = None if like is None else like.device
        projection = torch.tensor(self.projection, dtype=dtype, device=device).reshape(
            len(self.levels), self.rank
        )
        covariance = torch.tensor(self.covariance, dtype=dtype, device=device).reshape(
            self.rank, self.rank
        )
        factor = torch.tensor(self.sampling_factor, dtype=dtype, device=device).reshape(
            len(self.levels), self.rank
        )
        return projection, covariance, factor


def _tolerance(matrix: torch.Tensor) -> float:
    """Choose a relative Float64 eigensolver tolerance for a fixed matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Finite symmetric Double-precision matrix.

    Returns
    -------
    float
        Dimension-scaled backward-error threshold without an absolute unit floor.
    """
    scale = float(matrix.abs().max()) if matrix.numel() else 0.0
    return 64.0 * torch.finfo(torch.float64).eps * max(matrix.shape, default=1) * scale


__all__ = []
