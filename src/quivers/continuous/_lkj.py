"""The torch LKJ distribution over correlation Cholesky factors.

`OnionLKJCholesky` keeps the
log density, support, and parameter checks of
`torch.distributions.LKJCholesky` and draws by the onion construction in
`quivers.qiec._onion`, the implementation the reference machine also
draws by.
"""

from __future__ import annotations

import torch
import torch.distributions as D
from torch.types import _size

from quivers.qiec._onion import onion_cholesky_rows

__all__: list[str] = []


def _clamped_sqrt(value: torch.Tensor) -> torch.Tensor:
    """The square root, with its argument held at or above the smallest normal.

    Parameters
    ----------
    value : torch.Tensor
        A nonnegative tensor.

    Returns
    -------
    torch.Tensor
        ``sqrt(max(value, tiny))`` for the dtype's smallest normal number,
        so a Beta draw that rounds to one leaves a positive diagonal and
        the square root keeps a finite gradient.
    """
    return torch.clamp(value, min=torch.finfo(value.dtype).tiny).sqrt()


def lkj_cholesky_rsample(
    dimension: int,
    concentration: torch.Tensor,
    sample_shape: torch.Size = torch.Size(),
) -> torch.Tensor:
    """Draw LKJ correlation Cholesky factors by the onion method.

    The draw is the construction of
    `onion_cholesky_rows`
    (Lewandowski, Kurowicka, and Joe 2009, section 3.2,
    [doi:10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008))
    on tensors: each row's squared radius is a reparameterized
    `torch.distributions.Beta` draw and its direction is normalized
    standard normal noise. The factor is differentiable in the
    concentration through the Beta draws' implicit reparameterization;
    the directions do not depend on it.

    Parameters
    ----------
    dimension : int
        The matrix dimension, at least one.
    concentration : torch.Tensor
        The LKJ shape ``eta``, positive; its shape is the batch shape and
        its dtype and device are those of the draws.
    sample_shape : torch.Size
        Leading sample dimensions.

    Returns
    -------
    torch.Tensor
        Lower-triangular factors with positive diagonals and unit rows, of
        shape ``(*sample_shape, *concentration.shape, dimension,
        dimension)``.
    """
    shape = torch.Size(sample_shape) + concentration.shape

    def draw_beta(first: float, second: torch.Tensor) -> torch.Tensor:
        return D.Beta(torch.full_like(second, first), second).rsample(
            torch.Size(sample_shape)
        )

    def draw_direction(size: int) -> list[torch.Tensor]:
        noise = torch.randn(
            (size, *shape), dtype=concentration.dtype, device=concentration.device
        )
        return list(noise.unbind(0))

    rows = onion_cholesky_rows(
        dimension, concentration, draw_beta, draw_direction, _clamped_sqrt
    )
    zero = torch.zeros(shape, dtype=concentration.dtype, device=concentration.device)
    one = torch.ones(shape, dtype=concentration.dtype, device=concentration.device)
    lower = [[one], *rows]
    padded = [
        torch.stack(row + [zero] * (dimension - len(row)), dim=-1) for row in lower
    ]
    return torch.stack(padded, dim=-2)


class OnionLKJCholesky(D.LKJCholesky):
    """LKJ distribution over correlation Cholesky factors, drawn by the onion method.

    The parameters, support, and log density are those of
    `torch.distributions.LKJCholesky`. Draws come from
    `lkj_cholesky_rsample`,
    so every off-diagonal correlation ``r`` of a draw in dimension ``d``
    has ``(r + 1) / 2 ~ Beta(eta - 1 + d / 2, eta - 1 + d / 2)``, and
    ``rsample`` is differentiable in the concentration.

    Parameters
    ----------
    dim : int
        The matrix dimension, at least two.
    concentration : float or torch.Tensor
        The LKJ shape ``eta``, positive.
    validate_args : bool or None
        Whether to validate parameters and scored values.
    """

    has_rsample = True

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw factors with gradients to the concentration.

        Parameters
        ----------
        sample_shape : torch.Size or sequence of int
            Leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Factors of shape ``(*sample_shape, *batch_shape, dim, dim)``.
        """
        return lkj_cholesky_rsample(
            self.dim, self.concentration, torch.Size(sample_shape)
        )

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw factors without gradients.

        Parameters
        ----------
        sample_shape : torch.Size or sequence of int
            Leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Factors of shape ``(*sample_shape, *batch_shape, dim, dim)``.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)
