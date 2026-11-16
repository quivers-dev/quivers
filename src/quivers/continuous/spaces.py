"""Continuous measurable spaces for the hybrid architecture.

Continuous space objects serve as domains and codomains for continuous
morphisms. They complement [`quivers.core.objects`][quivers.core.objects] (FinSet etc.) used
by discrete morphisms.

The space family is a sum type:

- `Euclidean` — :math:`\\mathbb{R}^d` with optional bounds
- `Simplex` — probability simplex over ``d`` components
- `PositiveReals` — :math:`(0, \\infty)^d`
- `ProductSpace` — cartesian product of continuous spaces

`UnitInterval` is a convenience factory for ``[0, 1]^d``.
"""

from math import prod
from typing import Literal

import didactic.api as dx
import torch

from quivers.core.objects import SetObject


# ---------------------------------------------------------------------------
# the ContinuousSpace sum
# ---------------------------------------------------------------------------


class ContinuousSpace(dx.TaggedUnion, discriminator="kind"):
    """Continuous measurable space (Euclidean, Simplex, PositiveReals, Product).

    Variants expose ``name: str`` and ``dim: int`` either as fields (the
    atomic variants) or as derived properties (`ProductSpace`),
    plus a `contains` predicate over the support.
    """

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single sample."""
        return (self.dim,)  # type: ignore[attr-defined]

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check whether points lie in the support."""
        raise NotImplementedError(f"{type(self).__name__}.contains")

    def sample_uniform(self, n: int) -> torch.Tensor:
        """Sample n points uniformly from the space."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support uniform sampling"
        )

    def __mul__(self, other: "ContinuousSpace") -> "ProductSpace":
        if not isinstance(other, ContinuousSpace):
            return NotImplemented
        return ProductSpace(components=(self, other))


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------


class Euclidean(ContinuousSpace):
    """Euclidean space :math:`\\mathbb{R}^d`, optionally bounded.

    ``plate_rows`` marks the space as the flattened codomain of a
    plate draw: :math:`\\mathbb{R}^{d}` read as ``plate_rows``
    independent rows of width ``dim // plate_rows``. The distinction
    matters wherever a plate variable is consumed as a *parameter*
    rather than as a value, because the plate axis is a batch axis
    there and the per-row parameter width is `row_width`, never the
    flat ``dim``.

    ``intrinsic_shape`` optionally records the tensor axes of one event. Its
    product must equal ``dim``. Without it, events retain the legacy flat
    vector shape ``(dim,)``; neither a singleton axis nor a rectangle is
    inferred from the number of entries.

    Parameters
    ----------
    name : str
        Carrier label.
    dim : int
        Total scalar entries in one event.
    low : float or None
        Optional common lower bound on each entry.
    high : float or None
        Optional common upper bound on each entry.
    plate_rows : int or None
        Number of rows represented by a flattened plate carrier.
    intrinsic_shape : tuple of int or None
        Explicit tensor event axes, including any singleton or empty axes.
    kind : Literal["euclidean"]
        Tagged-union discriminator.
    """

    name: str
    dim: int
    low: float | None = None
    high: float | None = None
    plate_rows: int | None = None
    intrinsic_shape: tuple[int, ...] | None = None
    kind: Literal["euclidean"] = "euclidean"

    @dx.model_validator()
    def _validate_intrinsic_shape(self) -> None:
        """Require declared event axes to describe exactly the carrier entries.

        Raises
        ------
        ValueError
            If an extent is not a nonnegative integer or its product differs
            from the flat dimension.
        """
        if self.intrinsic_shape is not None:
            if any(type(size) is not int or size < 0 for size in self.intrinsic_shape):
                raise ValueError("intrinsic_shape extents must be nonnegative integers")
            if prod(self.intrinsic_shape) != self.dim:
                raise ValueError("intrinsic_shape product must equal Euclidean dim")

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Declared tensor event axes, or the legacy flat vector shape."""
        return (self.dim,) if self.intrinsic_shape is None else self.intrinsic_shape

    @property
    def is_bounded(self) -> bool:
        """Whether the space has finite bounds on all sides."""
        return self.low is not None and self.high is not None

    @property
    def row_width(self) -> int:
        """Width of a single row of the space.

        For an ordinary Euclidean space this is ``dim``. For a
        flattened plate codomain it is the per-row width, i.e.
        ``dim // plate_rows``.
        """
        rows = self.plate_rows
        if rows is None:
            return self.dim
        if rows <= 0 or self.dim % rows != 0:
            raise ValueError(
                f"Euclidean({self.name!r}, {self.dim}): plate_rows={rows} "
                f"does not divide the flat dimension {self.dim} into "
                f"whole rows"
            )
        return self.dim // rows

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check bounds over the event axes, retaining every batch axis.

        Parameters
        ----------
        x : torch.Tensor
            Points with the declared trailing event shape.

        Returns
        -------
        torch.Tensor
            Boolean membership with only the leading batch axes.

        Raises
        ------
        ValueError
            If an explicitly shaped point has incompatible event axes.
        """
        rank = len(self.event_shape)
        if self.intrinsic_shape is not None and (
            x.ndim < rank or rank and tuple(x.shape[-rank:]) != self.event_shape
        ):
            raise ValueError(
                f"expected trailing event shape {self.event_shape}, got {tuple(x.shape)}"
            )
        batch_shape = x.shape[:-rank] if rank else x.shape
        result = torch.ones(batch_shape, dtype=torch.bool, device=x.device)
        axes = tuple(range(x.ndim - rank, x.ndim))
        if self.low is not None:
            valid = x >= self.low
            result = result & (valid.all(dim=axes) if axes else valid)
        if self.high is not None:
            valid = x <= self.high
            result = result & (valid.all(dim=axes) if axes else valid)
        return result

    def sample_uniform(self, n: int) -> torch.Tensor:
        """Sample bounded events while retaining their declared tensor axes.

        Parameters
        ----------
        n : int
            Number of independently sampled events.

        Returns
        -------
        torch.Tensor
            Samples of shape ``(n, *event_shape)``.

        Raises
        ------
        ValueError
            If the space has no finite pair of bounds.
        """
        if not self.is_bounded:
            raise ValueError("cannot sample uniformly from unbounded Euclidean space")
        assert self.low is not None and self.high is not None
        return torch.rand(n, *self.event_shape) * (self.high - self.low) + self.low

    def __str__(self) -> str:
        bounds = ""
        if self.low is not None or self.high is not None:
            bounds = f", low={self.low}, high={self.high}"
        return f"Euclidean({self.name!r}, {self.dim}{bounds})"


def UnitInterval(name: str, dim: int = 1) -> Euclidean:
    """Create a :math:`[0, 1]^d` bounded Euclidean space."""
    return Euclidean(name=name, dim=dim, low=0.0, high=1.0)


# ---------------------------------------------------------------------------
# Simplex
# ---------------------------------------------------------------------------


class Simplex(ContinuousSpace):
    """The probability simplex :math:`\\{x \\in \\mathbb{R}^d : x_i \\geq 0, \\sum x_i = 1\\}`."""

    name: str
    dim: int
    kind: Literal["simplex"] = "simplex"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        nonneg = (x >= -1e-7).all(dim=-1)
        sums_to_one = (x.sum(dim=-1) - 1.0).abs() < 1e-5
        return nonneg & sums_to_one

    def sample_uniform(self, n: int) -> torch.Tensor:
        e = torch.distributions.Exponential(1.0).sample((n, self.dim))
        return e / e.sum(dim=-1, keepdim=True)

    def __str__(self) -> str:
        return f"Simplex({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# PositiveReals
# ---------------------------------------------------------------------------


class PositiveReals(ContinuousSpace):
    """The positive reals :math:`(0, \\infty)^d`."""

    name: str
    dim: int
    kind: Literal["positive_reals"] = "positive_reals"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0.0).all(dim=-1)

    def __str__(self) -> str:
        return f"PositiveReals({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# CholeskyFactor
# ---------------------------------------------------------------------------


class CholeskyFactor(ContinuousSpace):
    """The manifold of :math:`K \\times K` lower-triangular Cholesky factors.

    Each element is a lower-triangular matrix :math:`L` whose rows
    have unit norm: :math:`L_{ii}^2 + \\sum_{j<i} L_{ij}^2 = 1` for
    every :math:`i`. The product :math:`L L^T` is then a
    correlation matrix. The standard parameterization places
    :math:`L` on a :math:`K(K-1)/2`-dimensional manifold.

    Carrier represented as a flat :math:`K \\times K` array
    (row-major); the on-manifold constraint is enforced by the
    sampling family
    ([`quivers.continuous.families.LKJCorrelationFactor`][quivers.continuous.families.LKJCorrelationFactor])
    and not by the type itself.
    """

    name: str
    dim: int
    kind: Literal["cholesky_factor"] = "cholesky_factor"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# ProductSpace
# ---------------------------------------------------------------------------


def _flatten_spaces(
    items: tuple["ContinuousSpace | SetObject", ...],
) -> tuple["ContinuousSpace | SetObject", ...]:
    """Flatten nested ProductSpace so that P(A, P(B, C)) collapses to P(A, B, C)."""
    out: list[ContinuousSpace | SetObject] = []
    for s in items:
        if isinstance(s, ProductSpace):
            out.extend(s.components)
        else:
            out.append(s)
    return tuple(out)


def _component_name(c: "ContinuousSpace | SetObject") -> str:
    return c.name if isinstance(c, ContinuousSpace) else str(c)


def _component_dim(c: "ContinuousSpace | SetObject") -> int:
    """Return the event-shape width of a product component.

    ContinuousSpace components contribute `dim`; SetObject
    components contribute the length of their tensor `shape`
    (1 for FinSet, len(components) for ProductSet, 1 for CoproductSet).
    """
    return c.dim if isinstance(c, ContinuousSpace) else len(c.shape)


def _product_name(components: tuple["ContinuousSpace | SetObject", ...]) -> str:
    return " × ".join(_component_name(c) for c in components)


def _product_dim(components: tuple["ContinuousSpace | SetObject", ...]) -> int:
    return sum(_component_dim(c) for c in components)


class ProductSpace(ContinuousSpace):
    """Cartesian product of continuous spaces (and discrete objects).

    Components may be a mix of `ContinuousSpace` variants and
    [`quivers.core.objects.SetObject`][quivers.core.objects.SetObject] variants — programs whose
    domain or codomain combines discrete and continuous variables produce
    such a ProductSpace at compile time. Nested products are flattened
    on construction; `name` and `dim` are derived from
    `components` (for SetObject components, `dim` falls back
    to ``len(component.shape)``).
    """

    components: tuple[ContinuousSpace | SetObject, ...] = dx.field(
        default=(), converter=_flatten_spaces
    )
    kind: Literal["product_space"] = "product_space"

    @property
    def name(self) -> str:
        return _product_name(self.components)

    @property
    def dim(self) -> int:
        return _product_dim(self.components)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        offset = 0
        for s in self.components:
            width = _component_dim(s)
            if isinstance(s, ContinuousSpace):
                chunk = x[..., offset : offset + width]
                result = result & s.contains(chunk)
            else:
                # SetObject component (mixed-domain product): the slice is a
                # discrete index. Check it falls inside the index range.
                chunk = x[..., offset : offset + width]
                in_range = (chunk >= 0).all(dim=-1) & (chunk < s.size).all(dim=-1)
                result = result & in_range
            offset += width
        return result

    def __str__(self) -> str:
        inner = " × ".join(str(s) for s in self.components)
        return f"({inner})"


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------


class Sphere(ContinuousSpace):
    r"""The unit sphere :math:`S^{N-1} = \{x \in \mathbb{R}^N : \|x\|_2 = 1\}`.

    Ambient dimension is ``dim = N``; the manifold dimension is ``N - 1``.
    Carrier represented as an ``(N,)`` tensor; the unit-norm constraint
    is enforced by the sampling family (e.g. von Mises-Fisher),
    not by the type itself.
    """

    name: str
    dim: int
    kind: Literal["sphere"] = "sphere"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, dim=-1)
        return (norm - 1.0).abs() < 1e-5

    def sample_uniform(self, n: int) -> torch.Tensor:
        # Standard construction: normalized isotropic Gaussian.
        z = torch.randn(n, self.dim)
        return z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)

    def __str__(self) -> str:
        return f"Sphere({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------


class Ball(ContinuousSpace):
    r"""The closed ball :math:`\{x \in \mathbb{R}^N : \|x\|_2 \le r\}`.

    Carrier represented as an ``(N,)`` tensor. The ``radius`` field
    fixes ``r``; the default ``r = 1`` produces the unit ball.
    """

    name: str
    dim: int
    radius: float = 1.0
    kind: Literal["ball"] = "ball"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(x, dim=-1) <= self.radius

    def sample_uniform(self, n: int) -> torch.Tensor:
        # Uniform in the ball: sample direction from sphere, radius
        # from radius * U^{1/N} so volume is preserved.
        z = torch.randn(n, self.dim)
        direction = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True)
        u = torch.rand(n, 1)
        r = self.radius * u.pow(1.0 / self.dim)
        return direction * r

    def __str__(self) -> str:
        return f"Ball({self.name!r}, {self.dim}, radius={self.radius})"


# ---------------------------------------------------------------------------
# Covariance (symmetric positive-definite)
# ---------------------------------------------------------------------------


class Covariance(ContinuousSpace):
    r"""The cone of :math:`D \times D` symmetric positive-definite matrices.

    Used as the codomain of Wishart-shaped families. Carrier
    represented as a flat ``(D*D,)`` tensor (row-major); the
    positivity constraint is enforced by the sampling family.
    """

    name: str
    dim: int
    kind: Literal["covariance"] = "covariance"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        m = x.reshape(*x.shape[:-1], d, d)
        symmetric = (m - m.transpose(-1, -2)).abs().amax(dim=(-1, -2)) < 1e-5
        # All eigenvalues strictly positive (within float tolerance).
        eigvals = torch.linalg.eigvalsh(0.5 * (m + m.transpose(-1, -2)))
        positive = (eigvals > -1e-7).all(dim=-1)
        return symmetric & positive

    def __str__(self) -> str:
        return f"Covariance({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# Correlation (symmetric positive-definite with unit diagonal)
# ---------------------------------------------------------------------------


class Correlation(ContinuousSpace):
    r"""The manifold of :math:`D \times D` correlation matrices.

    A correlation matrix is symmetric positive-definite with unit
    diagonal: :math:`R_{ii} = 1` and :math:`R \succ 0`. Used as the
    codomain of LKJ correlation priors. Carrier represented as a
    flat ``(D*D,)`` tensor.
    """

    name: str
    dim: int
    kind: Literal["correlation"] = "correlation"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        m = x.reshape(*x.shape[:-1], d, d)
        symmetric = (m - m.transpose(-1, -2)).abs().amax(dim=(-1, -2)) < 1e-5
        diagonal_one = (torch.diagonal(m, dim1=-2, dim2=-1) - 1.0).abs().amax(
            dim=-1
        ) < 1e-5
        eigvals = torch.linalg.eigvalsh(0.5 * (m + m.transpose(-1, -2)))
        positive = (eigvals > -1e-7).all(dim=-1)
        return symmetric & diagonal_one & positive

    def __str__(self) -> str:
        return f"Correlation({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# Orthogonal
# ---------------------------------------------------------------------------


class Orthogonal(ContinuousSpace):
    r"""The orthogonal group :math:`O(D) = \{Q \in \mathbb{R}^{D \times D} : Q^T Q = I\}`.

    Carrier represented as a flat ``(D*D,)`` tensor. The orthogonality
    constraint is enforced by the sampling family; ``sample_uniform``
    produces Haar-distributed elements via QR decomposition of a
    standard Gaussian (the Mezzadri construction).
    """

    name: str
    dim: int
    kind: Literal["orthogonal"] = "orthogonal"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        m = x.reshape(*x.shape[:-1], d, d)
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        gram = m.transpose(-1, -2) @ m
        return (gram - eye).abs().amax(dim=(-1, -2)) < 1e-5

    def sample_uniform(self, n: int) -> torch.Tensor:
        # Mezzadri's algorithm: take QR of a standard Gaussian, then
        # fix the sign of each column of Q by the sign of the
        # corresponding diagonal entry of R. The result is Haar.
        z = torch.randn(n, self.dim, self.dim)
        q, r = torch.linalg.qr(z)
        d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        # Avoid zeros (probability zero in the Gaussian, but tolerate).
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = q * d.unsqueeze(-2)
        return q.reshape(n, self.dim * self.dim)

    def __str__(self) -> str:
        return f"Orthogonal({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# Stiefel
# ---------------------------------------------------------------------------


class Stiefel(ContinuousSpace):
    r"""The Stiefel manifold :math:`V_K(\mathbb{R}^N) = \{X \in \mathbb{R}^{N \times K} : X^T X = I_K\}`.

    Generalises `Orthogonal` to rectangular orthonormal-column
    matrices. ``rows`` is :math:`N`, ``cols`` is :math:`K`; require
    :math:`K \le N`. The base `dim` field is :math:`N`, kept
    for the standard ``event_shape = (dim,)`` contract; the actual
    carrier is flat with shape ``(N*K,)``.
    """

    name: str
    rows: int
    cols: int
    kind: Literal["stiefel"] = "stiefel"

    @property
    def dim(self) -> int:
        return self.rows

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows * self.cols,)

    @property
    def ndim(self) -> int:
        return 1

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        m = x.reshape(*x.shape[:-1], self.rows, self.cols)
        eye = torch.eye(self.cols, device=x.device, dtype=x.dtype)
        gram = m.transpose(-1, -2) @ m
        return (gram - eye).abs().amax(dim=(-1, -2)) < 1e-5

    def sample_uniform(self, n: int) -> torch.Tensor:
        # Haar-on-Stiefel via thin QR of a standard Gaussian (Mezzadri
        # again, restricted to the first K columns).
        if self.cols > self.rows:
            raise ValueError(f"Stiefel cols={self.cols} must be <= rows={self.rows}")
        z = torch.randn(n, self.rows, self.cols)
        q, r = torch.linalg.qr(z)
        d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = q * d.unsqueeze(-2)
        return q.reshape(n, self.rows * self.cols)

    def __str__(self) -> str:
        return f"Stiefel({self.name!r}, {self.rows}, {self.cols})"


# ---------------------------------------------------------------------------
# LowerTriangular
# ---------------------------------------------------------------------------


class LowerTriangular(ContinuousSpace):
    r"""The space of :math:`D \times D` lower-triangular matrices.

    The carrier holds a flat ``(D*D,)`` tensor; the structural zero
    constraint on the strictly upper-triangular entries is enforced
    by the sampling family / parameterization. Used as a
    parameterization carrier for Cholesky-style decompositions
    when no diagonal-sign constraint is required.
    """

    name: str
    dim: int
    kind: Literal["lower_triangular"] = "lower_triangular"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dim * self.dim,)

    @property
    def ndim(self) -> int:
        return 1

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        m = x.reshape(*x.shape[:-1], d, d)
        mask = torch.triu(torch.ones(d, d, device=x.device, dtype=torch.bool), 1)
        return (m * mask).abs().amax(dim=(-1, -2)) < 1e-7

    def __str__(self) -> str:
        return f"LowerTriangular({self.name!r}, {self.dim})"


# ---------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------


class Diagonal(ContinuousSpace):
    r"""The space of :math:`D \times D` diagonal matrices.

    Identified with :math:`\mathbb{R}^D` via the natural diagonal
    embedding; the carrier holds a flat ``(D,)`` tensor of diagonal
    entries. Used as the codomain of independent-variance
    parameterizations where each component carries its own scalar
    variance.
    """

    name: str
    dim: int
    kind: Literal["diagonal"] = "diagonal"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        # The (D,)-shaped carrier always satisfies the diagonality
        # constraint by construction; there's nothing to check on
        # the off-diagonals because there are none in the carrier.
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

    def sample_uniform(self, n: int) -> torch.Tensor:
        # No canonical "uniform" on the unbounded line; bail with a
        # clear error so users opt in to a specific prior family.
        raise NotImplementedError("Diagonal has no uniform measure on unbounded ℝ^D")

    def __str__(self) -> str:
        return f"Diagonal({self.name!r}, {self.dim})"


__all__ = [
    "Ball",
    "CholeskyFactor",
    "ContinuousSpace",
    "Correlation",
    "Covariance",
    "Diagonal",
    "Euclidean",
    "LowerTriangular",
    "Orthogonal",
    "PositiveReals",
    "ProductSpace",
    "Simplex",
    "Sphere",
    "Stiefel",
    "UnitInterval",
]
