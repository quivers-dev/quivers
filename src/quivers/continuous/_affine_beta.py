"""Affine Beta laws retaining both distances from their support endpoints."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import AffineTransform, Beta, TransformedDistribution

from quivers.continuous._beta_tails import _log_front, log_beta_tails
from quivers.continuous._simplex_sampling import _dirichlet_log_weights


class _IntervalExtension(AffineTransform):
    """Keep separate left/right extensions when evaluating an affine map.

    Parameters
    ----------
    left, right : Tensor
        Nonnegative distances extending the original unit interval.
    """

    def __init__(self, left: Tensor, right: Tensor):
        self.left, self.right = left, right
        self.width: Tensor = 1 + left + right
        super().__init__(-left, self.width)

    def _call(self, x: Tensor) -> Tensor:
        """Map a unit coordinate without subtracting large support offsets.

        Parameters
        ----------
        x : Tensor
            Unit-interval coordinate.

        Returns
        -------
        Tensor
            Affine response.
        """
        return x + (self.left + self.right) * (x - 0.5) + 0.5 * (self.right - self.left)

    def _inverse(self, y: Tensor) -> Tensor:
        """Invert the affine map with a centered path for large extensions.

        Parameters
        ----------
        y : Tensor
            Affine response.

        Returns
        -------
        Tensor
            Unit-interval coordinate.
        """
        ordinary = (y + self.left) / self.width
        centered = 0.5 + (y - 0.5 + 0.5 * (self.left - self.right)) / self.width
        return torch.where((self.left <= 1) & (self.right <= 1), ordinary, centered)


class _AffineBeta(TransformedDistribution):
    """Evaluate an extended Beta through its smaller endpoint coordinate.

    Parameters
    ----------
    concentration1, concentration0 : Tensor
        Positive Beta concentrations.
    left, right : Tensor
        Nonnegative extensions of the original unit interval.

    Notes
    -----
    The law is ``(1+left+right)*X-left``. Retaining both numerator
    coordinates avoids recovering a tiny upper-tail distance by subtracting
    a rounded transformed value from one.
    """

    _batch_parameter_ranks = {
        "concentration1": 0,
        "concentration0": 0,
        "left": 0,
        "right": 0,
    }

    def __init__(self, concentration1, concentration0, left, right):
        self.concentration1, self.concentration0, self.left, self.right = (
            torch.broadcast_tensors(
                concentration1,
                concentration0,
                left,
                right,
            )
        )
        self.extension = _IntervalExtension(self.left, self.right)
        super().__init__(Beta(self.concentration1, self.concentration0), self.extension)

    def expand(self, batch_shape, _instance=None):
        """Broadcast concentration and endpoint metadata together.

        Parameters
        ----------
        batch_shape : torch.Size
            Requested parameter batch axes.
        _instance : object or None
            Unused Torch expansion instance.

        Returns
        -------
        _AffineBeta
            Expanded affine law.
        """
        return type(self)(
            *(
                p.expand(batch_shape)
                for p in (
                    self.concentration1,
                    self.concentration0,
                    self.left,
                    self.right,
                )
            )
        )

    def _coordinates(self, value):
        """Return the smaller unit coordinate and correspondingly ordered shapes.

        Parameters
        ----------
        value : Tensor or float
            Response coordinate.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Small coordinate, its ordered shapes, and the reversal mask.
        """
        value = torch.as_tensor(value, dtype=self.left.dtype, device=self.left.device)
        lower = (value + self.left) / self.extension.width
        upper = (1 - value + self.right) / self.extension.width
        large = (self.left > 1) | (self.right > 1)
        lower = torch.where(
            large,
            0.5 + (value - 0.5 + 0.5 * (self.left - self.right)) / self.extension.width,
            lower,
        )
        upper = torch.where(
            large,
            0.5 + (0.5 - value + 0.5 * (self.right - self.left)) / self.extension.width,
            upper,
        )
        reverse = upper < lower
        point = torch.where(reverse, upper, lower)
        a = torch.where(reverse, self.concentration0, self.concentration1)
        b = torch.where(reverse, self.concentration1, self.concentration0)
        return point, a, b, reverse

    def log_prob(self, value):
        """Score the affine density without losing small endpoint distances.

        Parameters
        ----------
        value : Tensor or float
            Response coordinate.

        Returns
        -------
        Tensor
            Log density on broadcast sample and batch axes.
        """
        point, a, b, _ = self._coordinates(value)
        interior = (point > 0) & (point < 1)
        safe = torch.where(interior, point, 0.5)
        density = (
            _log_front(a, b, safe)
            - safe.log()
            - torch.log1p(-safe)
            - self.extension.width.log()
        )
        # Endpoint values follow the ordinary Beta density convention.
        endpoint = (
            torch.where(a == 1, b.log(), torch.where(a < 1, torch.inf, -torch.inf))
            - self.extension.width.log()
        )
        return torch.where(
            interior, density, torch.where(point == 0, endpoint, -torch.inf)
        )

    def _tails(self, value):
        """Evaluate complementary tails from the same smaller coordinate.

        Parameters
        ----------
        value : Tensor or float
            Response coordinate.

        Returns
        -------
        tuple[Tensor, Tensor]
            Complementary log CDF and log survival.
        """
        point, a, b, reverse = self._coordinates(value)
        cdf, sf = log_beta_tails(a, b, point)
        return torch.where(reverse, sf, cdf), torch.where(reverse, cdf, sf)

    def log_cdf(self, value):
        """Evaluate log CDF with paired endpoint coordinates.

        Parameters
        ----------
        value : Tensor or float
            Response coordinate.

        Returns
        -------
        Tensor
            Log cumulative probability.
        """
        return self._tails(value)[0]

    def log_survival_function(self, value):
        """Evaluate strict log survival with paired endpoint coordinates.

        Parameters
        ----------
        value : Tensor or float
            Response coordinate.

        Returns
        -------
        Tensor
            Log survival probability.
        """
        return self._tails(value)[1]

    def cdf(self, value):
        """Evaluate the affine Beta CDF.

        Parameters
        ----------
        value : Tensor or float
            Response coordinate.

        Returns
        -------
        Tensor
            Cumulative probability.
        """
        return self.log_cdf(value).exp()

    def _log_draw(self, sample_shape, *, reparameterized):
        """Keep complementary Beta weights before they round to an endpoint.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading independent sample axes.
        reparameterized : bool
            Use pathwise Gamma draws when true.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.
        """
        concentration = torch.stack((self.concentration1, self.concentration0), -1)
        return _dirichlet_log_weights(
            concentration.expand(torch.Size(sample_shape) + concentration.shape),
            reparameterized=reparameterized,
        )

    def _from_log_draw(self, weights):
        """Evaluate the affine map using the smaller complementary draw.

        Parameters
        ----------
        weights : Tensor
            Normalized log weights with a trailing pair of Beta coordinates.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.
        """
        x, q = weights[..., 0].exp(), weights[..., 1].exp()
        lower = (
            x + (self.left + self.right) * (x - 0.5) + 0.5 * (self.right - self.left)
        )
        upper = 1 - (
            q + (self.left + self.right) * (q - 0.5) + 0.5 * (self.left - self.right)
        )
        return torch.where(weights[..., 0] <= weights[..., 1], lower, upper)

    def rsample(self, sample_shape=torch.Size()):
        """Draw a pathwise affine Beta sample in paired endpoint coordinates.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading independent sample axes.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.
        """
        return self._from_log_draw(self._log_draw(sample_shape, reparameterized=True))

    def sample(
        self, sample_shape: torch.Size | tuple[int, ...] | list[int] = torch.Size()
    ):
        """Draw an affine Beta sample through the shared log-Gamma sampler.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading independent sample axes.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.

        Raises
        ------
        RuntimeError
            If the evaluation or sample exceeds the documented numerical capacity.
        """
        with torch.no_grad():
            return self._from_log_draw(
                self._log_draw(sample_shape, reparameterized=False)
            )

    def _sample_unit_censored(self, sample_shape, *, reparameterized):
        """Classify censoring atoms before rounding either Beta coordinate.

        Parameters
        ----------
        sample_shape : torch.Size or Sequence[int]
            Leading independent sample axes.
        reparameterized : bool
            Use pathwise Gamma draws when true.

        Returns
        -------
        Tensor
            Distribution result on the requested sample and batch axes.

        Raises
        ------
        RuntimeError
            If the evaluation or sample exceeds the documented numerical capacity.
        """
        weights = self._log_draw(sample_shape, reparameterized=reparameterized)
        lower = weights[..., 0] <= (self.left / self.extension.width).log()
        upper = weights[..., 1] <= (self.right / self.extension.width).log()
        value = self._from_log_draw(weights)
        interior = ~(lower | upper)
        if bool(
            (interior & ((value <= 0) | (value >= 1) | ~torch.isfinite(value))).any()
        ):
            raise RuntimeError(
                "affine Beta interior sample exceeds Float64 endpoint resolution"
            )
        return torch.where(lower, 0.0, torch.where(upper, 1.0, value))


__all__: list[str] = []
