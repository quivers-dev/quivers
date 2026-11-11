"""The onion construction of LKJ correlation Cholesky factors.

This module holds the single implementation of the LKJ sampler that both
runtimes share: the reference machine instantiates it on Python floats
with a :class:`random.Random`, and the compiled runtime instantiates it on
batched :class:`torch.Tensor` entries with reparameterized Beta draws.
Only the primitive draws and the square root differ between the two; the
row schedule and the assembly of each row are written once, here, so the
two runtimes cannot draw from different priors.

The construction is the onion method of Lewandowski, Kurowicka, and Joe
(2009, section 3.2), [doi:10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008),
for a correlation matrix with density proportional to
``det(R) ** (eta - 1)`` in dimension ``d``. Their algorithm starts from
``beta = eta + (d - 2) / 2`` and, for each ``m = 1, ..., d - 1`` (``m``
the size of the matrix built so far), lowers ``beta`` by one half after
the first step, draws ``y ~ Beta(m / 2, beta)`` and a direction ``u``
uniform on the unit sphere in ``R^m``, and extends the matrix by
``q = A w`` with ``w = sqrt(y) u`` and ``A`` the Cholesky factor of the
current matrix. Written on the Cholesky factor, the new row is
``(w, sqrt(1 - y))``, and ``beta`` at step ``m`` is
``eta + (d - 1 - m) / 2``. At ``m = 1`` this is the paper's
initialization, ``r_12 = 2 Beta(beta, beta) - 1``, since a symmetric
``r`` with ``r ** 2 ~ Beta(1/2, beta)`` has ``(r + 1) / 2 ~ Beta(beta,
beta)``. Every off-diagonal correlation thus has ``(r + 1) / 2 ~
Beta(eta - 1 + d / 2, eta - 1 + d / 2)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__: list[str] = []


def onion_cholesky_rows[S: (float, Tensor)](
    dimension: int,
    concentration: S,
    draw_beta: Callable[[float, S], S],
    draw_direction: Callable[[int], Sequence[S]],
    sqrt: Callable[[S], S],
) -> list[list[S]]:
    """Draw the rows below the first of an LKJ correlation Cholesky factor.

    Row ``m`` (zero-based, ``1 <= m < dimension``) of the factor is
    ``(sqrt(y) u, sqrt(1 - y))`` for ``y ~ Beta(m / 2, concentration +
    (dimension - 1 - m) / 2)`` and ``u`` uniform on the unit sphere in
    ``R^m``, the onion method of Lewandowski, Kurowicka, and Joe (2009,
    section 3.2), [doi:10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008).
    The direction is a vector of independent standard normals divided by
    its norm. Each row draws its Beta variate first and its direction
    second, in row order.

    Parameters
    ----------
    dimension : int
        The matrix dimension ``d``, at least one.
    concentration : float or torch.Tensor
        The LKJ shape ``eta``, a float or a tensor of per-draw shapes.
    draw_beta : Callable[[float, S], S]
        Draws ``Beta(a, b)`` for a first shape ``a`` and a second shape
        ``b`` of the concentration's kind.
    draw_direction : Callable[[int], Sequence[S]]
        Draws ``k`` independent standard normal coordinates.
    sqrt : Callable[[S], S]
        The square root on the scalar kind.

    Returns
    -------
    list[list[S]]
        For each row ``m = 1, ..., dimension - 1``, its ``m + 1`` entries
        on and below the diagonal: the ``m`` off-diagonal entries and then
        the diagonal entry. The first row, ``(1,)``, is left to the caller.
    """
    rows: list[list[S]] = []
    for m in range(1, dimension):
        radius_squared = draw_beta(m / 2.0, concentration + (dimension - 1 - m) / 2.0)
        direction = draw_direction(m)
        norm_squared = direction[0] * direction[0]
        for coordinate in direction[1:]:
            norm_squared = norm_squared + coordinate * coordinate
        norm = sqrt(norm_squared)
        radius = sqrt(radius_squared)
        row = [radius * coordinate / norm for coordinate in direction]
        row.append(sqrt(1.0 - radius_squared))
        rows.append(row)
    return rows
