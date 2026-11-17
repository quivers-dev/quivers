"""Retained Gaussian penalty coordinates and checked row-design contractions.

The design acts on coefficient coordinates; it does not fit a model or choose
priors. Dense spline bases and indexed group memberships share this boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from quivers.formulas._group_covariance import _matrix_values, _tolerance


@dataclass(frozen=True)
class DenseDesign:
    """A fixed row-by-coordinate map with no fitted coefficient state.

    Parameters
    ----------
    values : tuple[tuple[float, ...], ...]
        Finite known design values, copied into immutable Double coordinates.
    """

    values: tuple[tuple[float, ...], ...]
    coordinate_width: int | None = None

    def __post_init__(self) -> None:
        """Freeze numeric values and verify the retained coordinate width.

        Raises
        ------
        ValueError
            If the fixed matrix or explicit empty-design width is invalid.
        """
        object.__setattr__(self, "values", _matrix_values(self.values))
        width = len(self.values[0]) if self.values else self.coordinate_width
        if width is None or type(width) is not int or width < 0:
            raise ValueError("an empty dense design requires its coordinate width")
        if self.coordinate_width is not None and self.coordinate_width != width:
            raise ValueError("dense design width disagrees with its coordinates")
        object.__setattr__(self, "coordinate_width", width)

    @property
    def width(self) -> int:
        """Number of retained coefficient coordinates.

        Returns
        -------
        int
            Design width, including for empty response batches.
        """
        assert self.coordinate_width is not None
        return self.coordinate_width

    def apply(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Contract coordinates while preserving every leading sample axis.

        Parameters
        ----------
        coefficients : torch.Tensor
            Real coefficients with final axis equal to this design's width.

        Returns
        -------
        torch.Tensor
            Leading coefficient axes followed by the response-row axis.

        Raises
        ------
        ValueError
            If coefficient geometry or carrier disagrees with the design.
        """
        _check_coefficients(coefficients, self.width)
        matrix = coefficients.new_tensor(self.values).reshape(
            len(self.values), self.width
        )
        return coefficients @ matrix.T


@dataclass(frozen=True)
class PenaltyBasis:
    """Identified null and standardized Gaussian coordinates of one penalty.

    Parameters
    ----------
    width : int
        Original basis width.
    null_factor, random_factor : tuple[tuple[float, ...], ...]
        Original coefficient maps. The first spans the unpenalized subspace;
        the second has unit quadratic penalty, so a shared scale multiplies
        ordinary standard-Normal innovations.
    constraint : tuple[float, ...]
        Retained training-basis mean defining the sum-to-zero constraint.
        Empty means no centering constraint was requested.

    Notes
    -----
    Fresh rows use these factors and this training constraint unchanged.
    A null-space coefficient still needs its ordinary population prior; the
    Gaussian penalty alone does not define a proper prior in that direction.
    """

    width: int
    null_factor: tuple[tuple[float, ...], ...]
    random_factor: tuple[tuple[float, ...], ...]
    constraint: tuple[float, ...]

    @classmethod
    def fit(
        cls, design: torch.Tensor, penalty: torch.Tensor, *, center: bool = True
    ) -> PenaltyBasis:
        """Factor a positive-semidefinite penalty on its identified support.

        Parameters
        ----------
        design : torch.Tensor
            Fixed finite training basis, shape rows by coordinates.
        penalty : torch.Tensor
            Fixed symmetric positive-semidefinite coordinate penalty.
        center : bool
            Impose zero average fitted contribution over training rows.

        Returns
        -------
        PenaltyBasis
            Immutable retained coordinate transforms, without training rows.

        Raises
        ------
        ValueError
            If data, symmetry, penalty support, or null-space rank is invalid.
        """
        x = _fixed_tensor(design)
        s = _fixed_tensor(penalty)
        if not x.shape[0] or not x.shape[1]:
            raise ValueError("a penalty basis requires training rows and coordinates")
        width = x.shape[1]
        if s.shape != (width, width):
            raise ValueError(
                "penalty must have one row and column per basis coordinate"
            )
        tolerance = _tolerance(s)
        if not torch.allclose(s, s.T, atol=tolerance, rtol=0.0):
            raise ValueError("penalty must be symmetric")
        s = (s + s.T) * 0.5
        if bool((torch.linalg.eigvalsh(s) < -tolerance).any()):
            raise ValueError("penalty must be positive semidefinite")
        mean = x.mean(0)
        if center and bool((mean != 0).any()):
            q = torch.linalg.qr(mean[:, None], mode="complete").Q[:, 1:]
            constraint = tuple(mean.tolist())
        else:
            q = torch.eye(width, dtype=torch.float64)
            constraint = tuple(mean.tolist()) if center else ()
        restricted = q.T @ s @ q
        eigenvalues, eigenvectors = torch.linalg.eigh(restricted)
        tolerance = _tolerance(restricted)
        positive = eigenvalues > tolerance
        null = q @ eigenvectors[:, ~positive]
        random = q @ (eigenvectors[:, positive] / eigenvalues[positive].sqrt())
        # Null directions have no penalty to regularize an unidentified fit.
        # Their training design must therefore have its declared full rank.
        if null.shape[1] and int(torch.linalg.matrix_rank(x @ null)) != null.shape[1]:
            raise ValueError(
                "unpenalized basis directions are not identified by training rows"
            )
        return cls(
            width=width,
            null_factor=_matrix_values(null),
            random_factor=_matrix_values(random),
            constraint=constraint,
        )

    def evaluate(self, design: torch.Tensor) -> tuple[DenseDesign, DenseDesign]:
        """Apply fitted transforms to training or fresh basis evaluations.

        Parameters
        ----------
        design : torch.Tensor
            Fixed finite rows evaluated in the original retained basis.

        Returns
        -------
        tuple[DenseDesign, DenseDesign]
            Unpenalized and standardized Gaussian design maps.

        Raises
        ------
        ValueError
            If fresh basis width differs from its fitted width.
        """
        x = _fixed_tensor(design)
        if x.shape[1] != self.width:
            raise ValueError("fresh basis width differs from its fitted width")
        null = torch.tensor(self.null_factor, dtype=torch.float64)
        random = torch.tensor(self.random_factor, dtype=torch.float64)
        return DenseDesign(_matrix_values(x @ null), null.shape[1]), DenseDesign(
            _matrix_values(x @ random), random.shape[1]
        )


@dataclass(frozen=True)
class IndexedDesign:
    """A sparse weighted membership map into shared labeled coefficients.

    Parameters
    ----------
    levels : tuple[str, ...]
        Unique coefficient labels in their retained order.
    indices : tuple[tuple[int, ...], ...]
        One coordinate per member of each response row.
    weights : tuple[tuple[float, ...], ...]
        Finite weights aligned with indices. Duplicate members remain valid
        and contribute their combined weight without duplicating coefficients.
    """

    levels: tuple[str, ...]
    indices: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Validate sparse coordinate alignment and freeze fixed weights.

        Raises
        ------
        ValueError
            If labels, coordinate bounds, or row/member shapes disagree.
        """
        weights = _matrix_values(self.weights)
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError(
                "membership coefficient levels must be nonempty and unique"
            )
        if len(weights) != len(self.indices) or any(
            len(row) != len(weight)
            or any(
                type(index) is not int or not 0 <= index < len(self.levels)
                for index in row
            )
            for row, weight in zip(self.indices, weights, strict=True)
        ):
            raise ValueError(
                "membership coordinates and weights must align with retained levels"
            )
        object.__setattr__(self, "weights", weights)

    @classmethod
    def fit(
        cls,
        members: Sequence[Sequence[str]],
        weights: torch.Tensor | None = None,
        *,
        scale: bool = True,
        levels: tuple[str, ...] | None = None,
    ) -> IndexedDesign:
        """Bind membership rows without materializing a rows-by-levels matrix.

        Parameters
        ----------
        members : Sequence[Sequence[str]]
            Equal-width membership rows with at least two labeled members.
        weights : torch.Tensor or None
            Fixed real membership weights. None assigns equal weights.
        scale : bool
            Normalize nonnegative weights to sum to one per row. False permits
            signed weights and preserves their authored magnitudes.
        levels : tuple[str, ...] or None
            Retained coefficient labels for fresh rows, or None to fit their
            sorted union once from training memberships.

        Returns
        -------
        IndexedDesign
            Immutable sparse map, using O(rows times members) storage.

        Raises
        ------
        ValueError
            If labels, shapes, weight domains, or fitted levels disagree.
        """
        rows = tuple(tuple(row) for row in members)
        width = len(rows[0]) if rows else 0
        if width < 2 or any(len(row) != width for row in rows):
            raise ValueError(
                "multi-membership requires equally sized rows with at least two members"
            )
        if any(
            not isinstance(label, str) or not label for row in rows for label in row
        ):
            raise ValueError("membership labels must be nonempty strings")
        labels = (
            tuple(sorted({label for row in rows for label in row}))
            if levels is None
            else levels
        )
        if len(set(labels)) != len(labels) or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError(
                "retained membership levels must be unique nonempty strings"
            )
        lookup = {label: index for index, label in enumerate(labels)}
        missing = {label for row in rows for label in row} - lookup.keys()
        if missing:
            raise ValueError(f"unknown membership levels: {sorted(missing)!r}")
        values = (
            torch.ones((len(rows), width), dtype=torch.float64)
            if weights is None
            else torch.tensor(_matrix_values(weights), dtype=torch.float64)
        )
        if values.shape != (len(rows), width):
            raise ValueError("membership weights must align with every row and member")
        if scale:
            if bool((values < 0).any()):
                raise ValueError("negative membership weights require scale=False")
            total = values.sum(-1, keepdim=True)
            if bool((total <= 0).any()):
                raise ValueError(
                    "normalized membership weights need positive row totals"
                )
            # Divide before summing when finite authored magnitudes overflow.
            maximum = values.max(-1, keepdim=True).values
            values = values / maximum
            values = values / values.sum(-1, keepdim=True)
        return cls(
            labels,
            tuple(tuple(lookup[label] for label in row) for row in rows),
            _matrix_values(values),
        )

    def apply(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Gather shared coefficients and reduce only membership coordinates.

        Parameters
        ----------
        coefficients : torch.Tensor
            Coefficients with final axis in retained level order.

        Returns
        -------
        torch.Tensor
            Leading coefficient sample axes followed by response rows.

        Raises
        ------
        ValueError
            If coefficient carrier or final axis is invalid.
        """
        _check_coefficients(coefficients, len(self.levels))
        index = torch.tensor(self.indices, dtype=torch.long, device=coefficients.device)
        weights = coefficients.new_tensor(self.weights)
        return (coefficients[..., index] * weights).sum(-1)


def _check_coefficients(value: torch.Tensor, width: int) -> None:
    """Validate the shared trailing-coordinate contract of either design.

    Parameters
    ----------
    value : torch.Tensor
        Native coefficient values, retaining leading sample axes.
    width : int
        Required final coordinate extent.

    Raises
    ------
    ValueError
        If coefficient geometry or numeric carrier is invalid.
    """
    if (
        value.ndim < 1
        or value.shape[-1] != width
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise ValueError(
            "coefficient final axis must match the real design coordinates"
        )


def _fixed_tensor(value: torch.Tensor) -> torch.Tensor:
    """Reuse fixed-matrix validation while retaining empty tensor geometry.

    Parameters
    ----------
    value : torch.Tensor
        Fixed matrix to validate and copy.

    Returns
    -------
    torch.Tensor
        Independent Double-precision CPU matrix with its original shape.
    """
    values = _matrix_values(value)
    return torch.tensor(values, dtype=torch.float64).reshape(value.shape)


__all__ = []
