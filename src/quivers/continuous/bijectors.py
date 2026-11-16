"""Bijector library for the compositional measure algebra.

A `Bijector` is a measurable bijection $f: X \\to Y$ between Borel
spaces, together with its inverse $f^{-1}$ and the log-determinant
of its Jacobian. Pushing forward a measure $\\mu$ along $f$ produces
a new measure $f_*\\mu$ with density

$$p_{f_*\\mu}(y) = p_\\mu(f^{-1}(y))\\,\\bigl|\\det J_{f^{-1}}(y)\\bigr|.$$

In log-space (the only sane computational discipline; see
[TensorFlow Probability's `Bijector` rationale][tfp-bijector]):

$$\\log p_{f_*\\mu}(y) = \\log p_\\mu(f^{-1}(y)) + \\log\\bigl|\\det
J_{f^{-1}}(y)\\bigr|.$$

Every bijector exposes:

* `forward(x)` evaluating $f$,
* `inverse(y)` evaluating $f^{-1}$,
* `forward_log_det_jacobian(x)` evaluating
  $\\log\\bigl|\\det J_f(x)\\bigr|$,
* `inverse_log_det_jacobian(y)` evaluating
  $\\log\\bigl|\\det J_{f^{-1}}(y)\\bigr|$ (equal to
  $-\\text{forward\\_log\\_det\\_jacobian}(f^{-1}(y))$ when both are
  finite).

The library is closed under composition via `Compose(b1, b2)` and
under inversion via `Inverse(b)`, so the family forms a groupoid
of measurable isomorphisms.

[tfp-bijector]: https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Bijector
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import Tensor


def _extended_map(
    value: Tensor, mapping, bounds: tuple[float, float], image: tuple[float, float]
) -> Tensor:
    """Extend an increasing scalar map constantly beyond its domain.

    Parameters
    ----------
    value : Tensor
        Evaluation points.
    mapping : callable
        Interior increasing map.
    bounds : tuple[float, float]
        Domain bounds of that map.
    image : tuple[float, float]
        Corresponding image endpoints, including finite limits.

    Returns
    -------
    Tensor
        Mapped points, with the image limits beyond either domain boundary.
    """
    if bool(value.isnan().any()):
        raise ValueError("scalar interval bounds must not be NaN")
    low, high = bounds
    inside = (value > low) & (value < high) & torch.isfinite(value)
    anchor = (
        (low + high) / 2
        if math.isfinite(low) and math.isfinite(high)
        else low + 1
        if math.isfinite(low)
        else high - 1
        if math.isfinite(high)
        else 0.0
    )
    mapped = mapping(torch.where(inside, value, anchor))
    return torch.where(inside, mapped, torch.where(value <= low, image[0], image[1]))


class Bijector(ABC):
    """Abstract base for measurable bijections used in
    [`Pushforward`][quivers.continuous.measure.Pushforward].

    Implementations override the four primitive methods; the
    `__call__`, `Compose`, and `inv` helpers compose them.
    """

    _domain_bounds: tuple[float, float] | None = None
    _image_bounds: tuple[float, float] | None = None
    _batch_parameter_names: tuple[str, ...] | None = None
    jacobian_event_dim = 0

    def forward_shape(self, shape):
        """Retain intrinsic event axes for an elementwise transformation.

        Parameters
        ----------
        shape : torch.Size
            Input event shape.

        Returns
        -------
        torch.Size
            Output event shape.
        """
        return torch.Size(shape)

    def inverse_shape(self, shape):
        """Retain intrinsic event axes for the inverse elementwise map.

        Parameters
        ----------
        shape : torch.Size
            Output event shape.

        Returns
        -------
        torch.Size
            Input event shape.
        """
        return torch.Size(shape)

    def select_batch(self, batch_shape, active, *, event_shape=(), output_shape=None):
        """Select declared parameter batches while retaining intrinsic event axes.

        Parameters
        ----------
        batch_shape : torch.Size
            Broadcast batch of the measure before transformation.
        active : Tensor
            Boolean selection over that batch.
        event_shape : tuple[int, ...]
            Intrinsic coordinates to retain within each selected row.
        output_shape : torch.Size or None
            Optional shape for the selected batch axes.

        Returns
        -------
        Bijector
            Transformation with matching selected parameter rows.

        Raises
        ------
        NotImplementedError
            If a custom transformation declares no batch-selection capability.
        """
        if self._batch_parameter_names is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no batch-selection capability"
            )
        if not self._batch_parameter_names:
            return self
        from quivers.continuous._batch_selection import select_parameter

        parameters = {
            name: select_parameter(
                torch.as_tensor(getattr(self, name)).expand(
                    torch.Size(batch_shape) + torch.Size(event_shape)
                ),
                batch_shape,
                len(event_shape),
                active,
                output_shape=output_shape,
            )
            for name in self._batch_parameter_names
        }
        return type(self)(**parameters)

    def _interval_preimage(self, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
        """Pull back scalar bounds through a known increasing bijection.

        Parameters
        ----------
        low, high : Tensor
            Bounds in the codomain, including extended-real endpoints.

        Returns
        -------
        tuple[Tensor, Tensor]
            Extended-real bounds in the domain.

        Raises
        ------
        NotImplementedError
            If the bijector has no scalar interval capability.
        """
        if self._image_bounds is None or self._domain_bounds is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare a scalar interval preimage"
            )
        return (
            _extended_map(low, self.inverse, self._image_bounds, self._domain_bounds),
            _extended_map(high, self.inverse, self._image_bounds, self._domain_bounds),
        )

    def _interval_image(self, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
        """Map scalar bounds through a known increasing bijection.

        Parameters
        ----------
        low, high : Tensor
            Bounds in the domain, including extended-real endpoints.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bounds in the image.

        Raises
        ------
        NotImplementedError
            If no scalar domain contract is declared.
        """
        if self._domain_bounds is None or self._image_bounds is None:
            raise NotImplementedError(
                "interval images require a scalar domain contract"
            )
        return (
            _extended_map(low, self.forward, self._domain_bounds, self._image_bounds),
            _extended_map(high, self.forward, self._domain_bounds, self._image_bounds),
        )

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def inverse(self, y: Tensor) -> Tensor: ...

    @abstractmethod
    def forward_log_det_jacobian(self, x: Tensor) -> Tensor: ...

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        """Default implementation: invert the forward Jacobian's
        log-determinant. Subclasses with a more numerically stable
        closed form should override.
        """
        return -self.forward_log_det_jacobian(self.inverse(y))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def inv(self) -> Bijector:
        """Return the bijector's inverse, swapping `forward` /
        `inverse` and negating the Jacobian log-determinant.
        """
        return InverseBijector(self)


class InverseBijector(Bijector):
    """The inverse of a bijector, deferring all four primitives to
    the wrapped instance with their roles swapped.
    """

    def __init__(self, base: Bijector) -> None:
        self.base = base
        self.jacobian_event_dim = base.jacobian_event_dim

    def forward_shape(self, shape):
        """Use the retained bijector's inverse event-shape contract.

        Parameters
        ----------
        shape : torch.Size
            Input event shape.

        Returns
        -------
        torch.Size
            Output event shape under inversion.
        """
        return self.base.inverse_shape(shape)

    def inverse_shape(self, shape):
        """Use the retained bijector's forward event-shape contract.

        Parameters
        ----------
        shape : torch.Size
            Output event shape.

        Returns
        -------
        torch.Size
            Original input event shape.
        """
        return self.base.forward_shape(shape)

    def select_batch(self, batch_shape, active, *, event_shape=(), output_shape=None):
        """Select the wrapped transformation before taking its inverse.

        Parameters
        ----------
        batch_shape : torch.Size
            Retained measure batch.
        active : Tensor
            Boolean batch selection.
        event_shape : tuple[int, ...]
            Intrinsic event coordinates.
        output_shape : torch.Size or None
            Optional selected batch layout.

        Returns
        -------
        InverseBijector
            Inverse transformation on the selected batch.
        """
        return InverseBijector(
            self.base.select_batch(
                batch_shape,
                active,
                event_shape=self.base.inverse_shape(event_shape),
                output_shape=output_shape,
            )
        )

    def _interval_preimage(self, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
        """Use the base forward map when pulling back its inverse.

        Parameters
        ----------
        low, high : Tensor
            Bounds in the base domain.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bounds in the base image.

        Raises
        ------
        NotImplementedError
            If the base lacks an explicit scalar domain contract.
        """
        return self.base._interval_image(low, high)

    def _interval_image(self, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
        """Use the original preimage as the inverse bijector's image.

        Parameters
        ----------
        low, high : Tensor
            Scalar bounds in the inverse domain.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bounds in the inverse image.
        """
        return self.base._interval_preimage(low, high)

    def forward(self, x: Tensor) -> Tensor:
        return self.base.inverse(x)

    def inverse(self, y: Tensor) -> Tensor:
        return self.base.forward(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return self.base.inverse_log_det_jacobian(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return self.base.forward_log_det_jacobian(y)


class ComposeBijector(Bijector):
    """Composition `outer ∘ inner`. Forward applies `inner` first
    then `outer`; inverse applies `outer.inverse` first then
    `inner.inverse`. Log-Jacobians add by the chain rule.
    """

    def __init__(self, outer: Bijector, inner: Bijector) -> None:
        self.outer = outer
        self.inner = inner
        self.jacobian_event_dim = max(
            outer.jacobian_event_dim, inner.jacobian_event_dim
        )

    def forward_shape(self, shape):
        """Compose the intrinsic event-shape maps.

        Parameters
        ----------
        shape : torch.Size
            Original input event shape.

        Returns
        -------
        torch.Size
            Final output event shape.
        """
        return self.outer.forward_shape(self.inner.forward_shape(shape))

    def inverse_shape(self, shape):
        """Invert event-shape maps in the same order as coordinates.

        Parameters
        ----------
        shape : torch.Size
            Final output event shape.

        Returns
        -------
        torch.Size
            Original input event shape.
        """
        return self.inner.inverse_shape(self.outer.inverse_shape(shape))

    def select_batch(self, batch_shape, active, *, event_shape=(), output_shape=None):
        """Select both retained transformations under the same batch layout.

        Parameters
        ----------
        batch_shape : torch.Size
            Retained measure batch.
        active : Tensor
            Boolean batch selection.
        event_shape : tuple[int, ...]
            Intrinsic event coordinates.
        output_shape : torch.Size or None
            Optional selected batch layout.

        Returns
        -------
        ComposeBijector
            Composition of the selected transformations.
        """
        return ComposeBijector(
            self.outer.select_batch(
                batch_shape,
                active,
                event_shape=self.inner.forward_shape(event_shape),
                output_shape=output_shape,
            ),
            self.inner.select_batch(
                batch_shape, active, event_shape=event_shape, output_shape=output_shape
            ),
        )

    def _interval_preimage(self, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
        """Compose the same interval capability as the underlying scalar maps.

        Parameters
        ----------
        low, high : Tensor
            Bounds in the final image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bounds in the original domain.
        """
        return self.inner._interval_preimage(*self.outer._interval_preimage(low, high))

    def _interval_image(self, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
        """Compose images in the same order as the underlying forward maps.

        Parameters
        ----------
        low, high : Tensor
            Scalar bounds in the original domain.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bounds in the final image.
        """
        return self.outer._interval_image(*self.inner._interval_image(low, high))

    def forward(self, x: Tensor) -> Tensor:
        return self.outer.forward(self.inner.forward(x))

    def inverse(self, y: Tensor) -> Tensor:
        return self.inner.inverse(self.outer.inverse(y))

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        inner_x = self.inner.forward(x)
        return self._combine_jacobians(
            self.inner.forward_log_det_jacobian(x),
            self.outer.forward_log_det_jacobian(inner_x),
        )

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        outer_inv = self.outer.inverse(y)
        return self._combine_jacobians(
            self.inner.inverse_log_det_jacobian(outer_inv),
            self.outer.inverse_log_det_jacobian(y),
        )

    def _combine_jacobians(self, inner, outer):
        """Reduce only event axes that an elementwise member has not reduced.

        Parameters
        ----------
        inner, outer : Tensor
            The two log Jacobians under their own intrinsic event conventions.

        Returns
        -------
        Tensor
            Joint log Jacobian under the composition's event convention.
        """
        for value_name, member in (("inner", self.inner), ("outer", self.outer)):
            dims = self.jacobian_event_dim - member.jacobian_event_dim
            if dims:
                if value_name == "inner":
                    inner = inner.sum(tuple(range(-dims, 0)))
                else:
                    outer = outer.sum(tuple(range(-dims, 0)))
        return inner + outer


class IdentityBijector(Bijector):
    """The identity map. Useful as a neutral element in compositions
    and as the trivial pushforward.
    """

    _domain_bounds = (-math.inf, math.inf)
    _image_bounds = (-math.inf, math.inf)

    _batch_parameter_names = ()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, y: Tensor) -> Tensor:
        return y

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return torch.zeros_like(y)


class ExpBijector(Bijector):
    """The exponential map $f(x) = e^x$ from $\\mathbb{R}$ to
    $(0, \\infty)$. Jacobian log-determinant is $x$.
    """

    _domain_bounds = (-math.inf, math.inf)
    _image_bounds = (0.0, math.inf)

    _batch_parameter_names = ()

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.log(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return x

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(y)


class LogBijector(Bijector):
    """The logarithm $f(x) = \\log x$ from $(0, \\infty)$ to
    $\\mathbb{R}$. Inverse of [`ExpBijector`][quivers.continuous.bijectors.ExpBijector];
    Jacobian log-determinant is $-\\log x$.
    """

    _domain_bounds = (0.0, math.inf)
    _image_bounds = (-math.inf, math.inf)

    _batch_parameter_names = ()

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.exp(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.log(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return y


class SigmoidBijector(Bijector):
    """The logistic sigmoid $f(x) = 1/(1 + e^{-x})$ from
    $\\mathbb{R}$ to $(0, 1)$.

    The Jacobian log-determinant is the log-density of the standard
    logistic distribution, $-x - 2\\log(1 + e^{-x})$, in a form that
    is stable in both tails.
    """

    _domain_bounds = (-math.inf, math.inf)
    _image_bounds = (0.0, 1.0)

    _batch_parameter_names = ()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.logit(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(x) - torch.nn.functional.softplus(-x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(y) - torch.log1p(-y)


class LogitBijector(Bijector):
    """The logit $f(x) = \\log(x / (1 - x))$ from $(0, 1)$ to
    $\\mathbb{R}$. Inverse of [`SigmoidBijector`][quivers.continuous.bijectors.SigmoidBijector].
    """

    _domain_bounds = (0.0, 1.0)
    _image_bounds = (-math.inf, math.inf)

    _batch_parameter_names = ()

    def forward(self, x: Tensor) -> Tensor:
        return torch.logit(x)

    def inverse(self, y: Tensor) -> Tensor:
        return torch.sigmoid(y)

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.log(x) - torch.log1p(-x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(y) - torch.nn.functional.softplus(-y)


class SoftplusBijector(Bijector):
    """The softplus map $f(x) = \\log(1 + e^x)$ from $\\mathbb{R}$
    to $(0, \\infty)$. Smooth alternative to
    [`ExpBijector`][quivers.continuous.bijectors.ExpBijector] with linear tail growth
    on the positive side.

    Forward Jacobian: $df/dx = \\sigma(x)$, so
    $\\log|df/dx| = -\\text{softplus}(-x)$.

    Inverse: $f^{-1}(y) = \\log(e^y - 1)$.
    Inverse Jacobian: $df^{-1}/dy = 1/(1 - e^{-y})$, so
    $\\log|df^{-1}/dy| = -\\log(1 - e^{-y}) = -\\log(-\\mathrm{expm1}(-y))$.
    """

    _domain_bounds = (-math.inf, math.inf)
    _image_bounds = (0.0, math.inf)

    _batch_parameter_names = ()

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.softplus(x)

    def inverse(self, y: Tensor) -> Tensor:
        return y + torch.log(-torch.expm1(-y))

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(-x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        return -torch.log(-torch.expm1(-y))


class AffineBijector(Bijector):
    """The affine map $f(x) = \\text{scale} \\cdot x + \\text{shift}$.
    `scale` must be strictly positive; the Jacobian log-determinant
    is $\\log\\,\\text{scale}$, broadcast to the input shape.
    """

    _domain_bounds = (-math.inf, math.inf)
    _image_bounds = (-math.inf, math.inf)

    _batch_parameter_names = ("scale", "shift")

    def __init__(
        self,
        scale: Tensor | float,
        shift: Tensor | float,
    ) -> None:
        self.scale = torch.as_tensor(scale)
        self.shift = torch.as_tensor(shift)
        if torch.any(self.scale <= 0):
            raise ValueError(
                f"Affine: scale must be strictly positive; got {self.scale!r}"
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.shift

    def inverse(self, y: Tensor) -> Tensor:
        return (y - self.shift) / self.scale

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        log_scale = torch.log(self.scale)
        return log_scale.expand_as(x)

    def inverse_log_det_jacobian(self, y: Tensor) -> Tensor:
        log_scale = torch.log(self.scale)
        return (-log_scale).expand_as(y)


class StickBreakingBijector(Bijector):
    """The stick-breaking map from $\\mathbb{R}^{K-1}$ to the
    open $K$-simplex $\\{p \\in \\mathbb{R}^K_{>0} : \\sum_k p_k = 1\\}$.

    Forward: $y_k = z_k \\cdot \\prod_{j<k}(1 - z_j)$ for
    $k < K-1$, where $z_k = \\sigma(x_k - \\log(K - k - 1))$, and
    $y_{K-1} = 1 - \\sum_{k<K-1} y_k$. The shift by
    $\\log(K - k - 1)$ centres the prior on the uniform simplex
    when $x \\sim \\mathrm{Normal}(0, 1)$, matching the convention
    used by Stan and PyMC.

    Inverse: $x_k = \\sigma^{-1}(z_k) + \\log(K - k - 1)$ where
    $z_k = y_k / (1 - \\sum_{j<k} y_j)$.

    The Jacobian log-determinant in the forward direction sums
    `log(z_k) + log(1 - z_k) + log(remaining)` across the K-1
    stick-breaks. Used in the desugaring of `LogisticNormal` and
    other simplex-valued transforms.
    """

    _batch_parameter_names = ()
    jacobian_event_dim = 1

    def forward_shape(self, shape):
        """Add the final simplex coordinate to a vector event.

        Parameters
        ----------
        shape : torch.Size
            Real-coordinate vector event shape.

        Returns
        -------
        torch.Size
            Simplex event shape with one additional coordinate.

        Raises
        ------
        ValueError
            If the input has no intrinsic vector axis.
        """
        if not shape:
            raise ValueError("stick breaking requires an intrinsic vector event axis")
        return torch.Size((*shape[:-1], shape[-1] + 1))

    def inverse_shape(self, shape):
        """Remove the simplex's redundant coordinate from its event shape.

        Parameters
        ----------
        shape : torch.Size
            Simplex event shape.

        Returns
        -------
        torch.Size
            Real-coordinate event shape.

        Raises
        ------
        ValueError
            If no nonempty simplex coordinate axis is supplied.
        """
        if not shape or shape[-1] < 1:
            raise ValueError("inverse stick breaking requires a simplex event axis")
        return torch.Size((*shape[:-1], shape[-1] - 1))

    def forward(self, x: Tensor) -> Tensor:
        K_minus_1 = x.shape[-1]
        offsets = torch.log(
            torch.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        )
        z = torch.sigmoid(x - offsets)
        log_remaining = torch.log1p(-z).cumsum(dim=-1)
        first = z[..., :1]
        rest = z[..., 1:] * torch.exp(log_remaining[..., :-1])
        y_first_K_minus_1 = torch.cat([first, rest], dim=-1)
        y_last = (1.0 - y_first_K_minus_1.sum(dim=-1, keepdim=True)).clamp_min(
            torch.finfo(x.dtype).tiny
        )
        return torch.cat([y_first_K_minus_1, y_last], dim=-1)

    def inverse(self, y: Tensor) -> Tensor:
        K = y.shape[-1]
        K_minus_1 = K - 1
        offsets = torch.log(
            torch.arange(K_minus_1, 0, -1, dtype=y.dtype, device=y.device)
        )
        cumulative = y[..., :K_minus_1].cumsum(dim=-1)
        first = y[..., :1]
        denom_rest = (1.0 - cumulative[..., :-1]).clamp_min(torch.finfo(y.dtype).tiny)
        rest = y[..., 1:K_minus_1] / denom_rest
        z = torch.cat([first, rest], dim=-1)
        return (
            torch.logit(
                z.clamp(
                    torch.finfo(y.dtype).tiny,
                    1.0 - torch.finfo(y.dtype).tiny,
                )
            )
            + offsets
        )

    def forward_log_det_jacobian(self, x: Tensor) -> Tensor:
        K_minus_1 = x.shape[-1]
        offsets = torch.log(
            torch.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        )
        shifted = x - offsets
        log_z = torch.nn.functional.logsigmoid(shifted)
        log_1m_z = torch.nn.functional.logsigmoid(-shifted)
        cum_log_1m_z = log_1m_z.cumsum(dim=-1)
        leading = cum_log_1m_z[..., :-1].sum(dim=-1)
        return log_z.sum(dim=-1) + log_1m_z.sum(dim=-1) + leading


BIJECTORS_BY_NAME: Mapping[str, type[Bijector]] = MappingProxyType(
    {
        "Identity": IdentityBijector,
        "Exp": ExpBijector,
        "Log": LogBijector,
        "Sigmoid": SigmoidBijector,
        "Logit": LogitBijector,
        "Softplus": SoftplusBijector,
        "StickBreaking": StickBreakingBijector,
    }
)
"""The bijectors a QVR source names by their surface spelling.

``Pushforward(base, Exp)`` resolves ``Exp`` here. Only bijectors that
take no arguments are listed; an affine or composite map is built in
Python and passed as a value.
"""


__all__ = [
    "BIJECTORS_BY_NAME",
    "Bijector",
    "AffineBijector",
    "ComposeBijector",
    "ExpBijector",
    "IdentityBijector",
    "InverseBijector",
    "LogBijector",
    "LogitBijector",
    "SigmoidBijector",
    "SoftplusBijector",
    "StickBreakingBijector",
]
