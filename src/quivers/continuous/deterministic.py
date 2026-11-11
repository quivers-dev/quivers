"""Deterministic morphisms in :math:`\\mathbf{Kern}`.

Each builder returns a `ContinuousMorphism` whose runtime
realisation is a Dirac kernel concentrated on a deterministic
image: ``cumsum_morphism`` for ordinal monotone splines,
``softmax_morphism`` for the standard simplex projection,
``cholesky_quad_form_morphism`` for
covariance reconstruction from a correlation Cholesky factor and
a positive-scale vector.

Categorically a deterministic morphism :math:`f : X \\to Y` is the
unit-Kleisli image of a measurable map; its log-density is
:math:`-\\infty` everywhere except at :math:`y = f(x)` and zero
there. The variational backend treats the sampling path as an
identity reparameterization of the deterministic image and the
log-density as a no-op (the surrounding KL terms absorb the
Dirac indicator).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.continuous.param_source import ParamSource
from quivers.continuous.spaces import (
    CholeskyFactor,
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
)

type FunctionOutput = Literal["identity", "tanh", "sigmoid", "softplus", "relu"]
"""The nonlinearity a `LearnedFunction` applies to its network's output."""

_OUTPUTS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda value: value,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softplus": nn.functional.softplus,
    "relu": torch.relu,
}

#: The output nonlinearities a `LearnedFunction` accepts, by name.
FUNCTION_OUTPUTS: frozenset[str] = frozenset(_OUTPUTS)


class DeterministicMorphism(ContinuousMorphism):
    """Helper base for deterministic morphisms ``f : X → Y``.

    A deterministic morphism in :math:`\\mathbf{Kern}` is a Dirac
    kernel concentrated on :math:`f(x)`. Its log-density is
    :math:`-\\infty` everywhere except at :math:`y = f(x)` and
    :math:`0` (= :math:`\\log 1`) there; for gradient-based inference
    we treat the sampling path as an identity reparameterization
    of the deterministic image.

    Subclasses supply ``_apply(x) -> y``.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        apply: Callable[[torch.Tensor], torch.Tensor],
        name: str,
    ) -> None:
        super().__init__(domain, codomain)
        self._apply_fn = apply
        self._name = name

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        del sample_shape
        return self._apply_fn(x)

    def point_mass_value(self, x: torch.Tensor) -> torch.Tensor:
        """The Dirac's location, :math:`f(x)`.

        Reporting it lets a composition through this morphism
        integrate over its codomain exactly, by evaluating the next
        kernel at the single point carrying all the mass, instead of
        approximating an integral that has no spread to resolve.
        """
        return self._apply_fn(x)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Deterministic kernels: log-prob is 0 at y == f(x), -inf otherwise.
        # In the variational backend we drop the indicator and treat
        # the morphism's effective log-density as zero (it's a Dirac
        # kernel; the inference target's gradient is uninformed by
        # this density and the surrounding KL terms absorb it).
        expected = self._apply_fn(x)
        return torch.zeros(expected.shape[:1], device=x.device, dtype=x.dtype)

    def __repr__(self) -> str:
        return f"{self._name}({self.domain!s} → {self.codomain!s})"


def cumsum_morphism(dim: int) -> DeterministicMorphism:
    """Cumulative sum ``cumsum : Euclidean(dim) → Euclidean(dim)``.

    Maps :math:`(x_1, \\dots, x_K)` to
    :math:`(x_1, x_1 + x_2, \\dots, \\sum_i x_i)`. The canonical
    monotonic-spline parameterization for ordinal covariates
    (``coef ~ Normal(0, sigma)`` followed by ``cumsum(coef - coef[0])``
    gives a 0-anchored, monotonically-increasing effect across the
    ordered levels).
    """
    space = Euclidean(name="cumsum", dim=dim)
    return DeterministicMorphism(
        space, space, lambda x: torch.cumsum(x, dim=-1), name="cumsum"
    )


def softmax_morphism(dim: int) -> DeterministicMorphism:
    """Softmax ``softmax : Euclidean(dim) → Simplex(dim)``.

    The standard exponential normalizer onto the probability
    simplex. Useful for closed-form class-probability computations
    in posterior blocks where one would otherwise need a
    log-sum-exp aggregation by hand.
    """
    src = Euclidean(name="softmax_in", dim=dim)
    tgt = Simplex(name="softmax_out", dim=dim)
    return DeterministicMorphism(
        src, tgt, lambda x: torch.softmax(x, dim=-1), name="softmax"
    )


def cholesky_quad_form_morphism(dim: int) -> ContinuousMorphism:
    """Covariance reconstruction ``(L, s) -> diag(s) L L^T diag(s)``.

    Given a Cholesky factor :math:`L` of a :math:`K \\times K`
    correlation matrix and a positive-scale vector
    :math:`s \\in (0, \\infty)^K`, returns the corresponding
    covariance matrix :math:`\\Sigma = D R D^T` with
    :math:`R = L L^T` and :math:`D = \\mathrm{diag}(s)`.

    Domain is the product
    `CholeskyFactor(K) <quivers.continuous.spaces.CholeskyFactor>` ``*``
    `PositiveReals(K) <quivers.continuous.spaces.PositiveReals>`;
    codomain is `Euclidean(K * K) <quivers.continuous.spaces.Euclidean>`
    flattened in row-major order so the result composes with
    downstream
    [`quivers.continuous.families.ConditionalMultivariateNormal`][quivers.continuous.families.ConditionalMultivariateNormal]
    consumers that accept a flat covariance vector.
    """
    cholesky = CholeskyFactor(name="L", dim=dim)
    scale = PositiveReals(name="scale", dim=dim)
    source = ProductSpace(components=(cholesky, scale))
    target = Euclidean(name="cov", dim=dim * dim)

    def _apply(xs: torch.Tensor) -> torch.Tensor:
        # xs : (batch, K*K + K) — the cholesky's flat lower-tri plus
        # the K-vector of scales, concatenated in that order. We
        # rehydrate L from the lower-triangular packing, apply the
        # quadratic form, then flatten.
        batch = xs.shape[0]
        chol_flat = xs[:, : dim * dim]
        scale_vec = xs[:, dim * dim :]
        L = chol_flat.reshape(batch, dim, dim)
        mask = torch.tril(torch.ones(dim, dim, device=xs.device, dtype=xs.dtype))
        L = L * mask
        R = L @ L.transpose(-1, -2)
        D = scale_vec.unsqueeze(-1) * torch.eye(dim, device=xs.device, dtype=xs.dtype)
        cov = D @ R @ D
        return cov.reshape(batch, dim * dim)

    return DeterministicMorphism(source, target, _apply, name="cholesky_quad_form")


def _event_width(space: AnySpace) -> int:
    """The number of real coordinates one value of a space carries.

    Parameters
    ----------
    space : AnySpace
        The space.

    Returns
    -------
    int
        Its width.

    Raises
    ------
    TypeError
        If the space is not continuous.
    """
    if not isinstance(space, ContinuousSpace):
        raise TypeError(
            f"LearnedFunction: a learned function maps real coordinates, but "
            f"{space!r} is not a continuous space"
        )
    return int(space.dim)


def function_factor_widths(space: AnySpace) -> tuple[int, ...]:
    """The width of each factor of a learned function's domain.

    Parameters
    ----------
    space : AnySpace
        The domain; a `ProductSpace` lists its factors in order.

    Returns
    -------
    tuple[int, ...]
        One width per factor, the one width of a space that is not a
        product.

    Raises
    ------
    TypeError
        If a factor is not continuous.
    """
    if isinstance(space, ProductSpace):
        return tuple(_event_width(component) for component in space.components)
    return (_event_width(space),)


class LearnedFunction(DeterministicMorphism):
    """A learned deterministic map ``f : X -> Y`` between continuous spaces.

    The map applies a parameter source to the input's coordinates and
    then an output nonlinearity, :math:`f(x) = \\sigma(g_\\theta(x))`,
    with :math:`g_\\theta` a [`LinearSource`][quivers.continuous.param_source.LinearSource]
    (the affine map :math:`W x + b`), an
    [`MLPSource`][quivers.continuous.param_source.MLPSource], or an
    [`IdentitySource`][quivers.continuous.param_source.IdentitySource].
    As a kernel it is the Dirac kernel at :math:`f(x)`: it contributes
    no density and no site, a composition through it is exact (see
    [`point_mass_value`][quivers.continuous.deterministic.DeterministicMorphism.point_mass_value]),
    and a scan over it is a deterministic recurrence. Its weights are
    module parameters, fit as point estimates by the objective that
    trains the model around it.

    Parameters
    ----------
    domain : AnySpace
        The input space. A `ProductSpace` takes its factors in order,
        concatenated into one coordinate vector.
    codomain : AnySpace
        The output space, as wide as the source's output.
    source : ParamSource
        The network from input coordinates to output coordinates.
    output : FunctionOutput
        The nonlinearity applied to the network's output.

    Raises
    ------
    TypeError
        If the domain or codomain is not continuous.
    ValueError
        If the source's output width is not the codomain's width, or
        ``output`` is not a `FunctionOutput`.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        source: ParamSource,
        output: FunctionOutput = "identity",
    ) -> None:
        widths = function_factor_widths(domain)
        width = _event_width(codomain)
        if int(source.param_dim) != width:
            raise ValueError(
                f"LearnedFunction: the source writes {source.param_dim} "
                f"coordinates, but the codomain {codomain!s} is {width} wide"
            )
        if output not in _OUTPUTS:
            raise ValueError(
                f"LearnedFunction: unknown output nonlinearity {output!r}; "
                f"expected one of {sorted(_OUTPUTS)}"
            )
        super().__init__(domain, codomain, self.evaluate, name="LearnedFunction")
        self.source = source
        self._output: FunctionOutput = output
        self._factor_widths = widths
        self._width = width

    @property
    def output(self) -> FunctionOutput:
        """The nonlinearity applied to the network's output.

        Returns
        -------
        FunctionOutput
            Its name.
        """
        return self._output

    @property
    def factor_widths(self) -> tuple[int, ...]:
        """The width of each factor of the domain, in order.

        Returns
        -------
        tuple[int, ...]
            One width per factor.
        """
        return self._factor_widths

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the map to a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor
            Inputs, shape ``(batch, domain_dim)``, or ``(batch,)`` for a
            one-wide domain; any leading axes before the coordinate axis
            are kept.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, codomain_dim)``, with the leading axes of
            ``x``.

        Raises
        ------
        ValueError
            If the coordinate axis is not the domain's width.
        """
        dimension = sum(self._factor_widths)
        if x.dim() == 1 and dimension == 1:
            x = x.unsqueeze(-1)
        if x.dim() == 0 or int(x.shape[-1]) != dimension:
            raise ValueError(
                f"LearnedFunction: expected inputs whose last axis is the "
                f"domain's {dimension} coordinates, got shape {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            x = x.to(torch.get_default_dtype())
        lead = x.shape[:-1]
        flat = x.reshape(-1, dimension)
        image = _OUTPUTS[self._output](self.source(flat))
        return image.reshape(*lead, self._width)

    def apply_values(self, *values: torch.Tensor) -> torch.Tensor:
        """Apply the map to one value per domain factor.

        This is how a program's ``let y = f(a, b)`` calls the map: each
        argument is one factor's value, a scalar for a one-wide factor
        or a vector of the factor's width, possibly led by axes the
        arguments share.

        Parameters
        ----------
        *values : torch.Tensor
            One value per factor, in the domain's order.

        Returns
        -------
        torch.Tensor
            The image, shape ``(*lead, codomain_dim)`` for the arguments'
            shared leading axes ``lead``.

        Raises
        ------
        ValueError
            If the number of values is not the number of factors, or a
            value's trailing axis is not its factor's width.
        """
        if len(values) != len(self._factor_widths):
            raise ValueError(
                f"LearnedFunction: the domain has {len(self._factor_widths)} "
                f"factors, but the call passes {len(values)} arguments"
            )
        shaped: list[torch.Tensor] = []
        for position, (value, width) in enumerate(
            zip(values, self._factor_widths, strict=True)
        ):
            tensor = torch.as_tensor(value)
            if not tensor.is_floating_point():
                tensor = tensor.to(torch.get_default_dtype())
            if tensor.dim() == 0 and width == 1:
                tensor = tensor.reshape(1)
            if tensor.dim() == 0 or int(tensor.shape[-1]) != width:
                raise ValueError(
                    f"LearnedFunction: argument {position} has shape "
                    f"{tuple(tensor.shape)}, but its domain factor is "
                    f"{width} wide"
                )
            shaped.append(tensor)
        lead = torch.broadcast_shapes(*(tensor.shape[:-1] for tensor in shaped))
        joined = torch.cat(
            [tensor.expand(*lead, tensor.shape[-1]) for tensor in shaped], dim=-1
        )
        return self.evaluate(joined)

    def __repr__(self) -> str:
        return (
            f"LearnedFunction({self.domain!s} → {self.codomain!s}, "
            f"source={type(self.source).__name__}, output={self._output})"
        )


__all__ = [
    "DeterministicMorphism",
    "FUNCTION_OUTPUTS",
    "FunctionOutput",
    "LearnedFunction",
    "function_factor_widths",
    "cholesky_quad_form_morphism",
    "cumsum_morphism",
    "softmax_morphism",
]
