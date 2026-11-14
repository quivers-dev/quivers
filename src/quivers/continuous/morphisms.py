"""Markov kernels on discrete, continuous, and mixed spaces.

``ContinuousMorphism`` defines ``log_prob`` and ``rsample``. The ``>>``
operator composes kernels and ``@`` forms their independent product.
Discrete intermediates are marginalized by finite summation. Continuous
intermediates are scored along the deterministic reference path described
by ``SampledComposition.log_prob``.
"""

from __future__ import annotations

import collections.abc
import math
from abc import ABC, abstractmethod
from typing import cast

import torch
import torch.nn as nn
from torch.distributions import constraints as _constraints

from quivers.continuous.spaces import ContinuousSpace, Euclidean, ProductSpace
from quivers.core.morphisms import Morphism
from quivers.core.morphisms import Morphism as _CatMorphism
from quivers.core.objects import SetObject

type AnySpace = SetObject | ContinuousSpace
"""A space a morphism maps between: a finite set or a continuous space."""

_QUANTILE_EPS = 1e-12
"""Clamp keeping the standard-normal quantile function finite.

Sobol coordinates land on dyadic rationals in ``[0, 1)``, and the
quantile function diverges at both ends; the clamp bounds the extreme
node at roughly seven standard deviations, far outside the region any
finite point set resolves."""


def _is_discrete(space: AnySpace) -> bool:
    """Check whether a space is discrete (SetObject)."""
    return isinstance(space, SetObject)


def _next_power_of_two(count: int) -> int:
    """Smallest power of two at least ``count`` (and at least one)."""
    if count <= 1:
        return 1
    return 1 << (count - 1).bit_length()


def dimension_probe(x: torch.Tensor) -> torch.Tensor:
    """A one-row slice of ``x``, enough to settle coordinate counts.

    A morphism's
    [`base_dimension`][quivers.continuous.morphisms.ContinuousMorphism.base_dimension]
    depends on trailing event extents, not the number of rows.
    """
    return x[:1]


def _event_size(shape: torch.Size) -> int:
    """Product of a ``(batch, *event)`` shape's trailing axes."""
    size = 1
    for extent in shape[1:]:
        size *= int(extent)
    return size


def sobol_normal_points(
    dimension: int,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A deterministic standard-normal point set of shape ``(n, dimension)``.

    Push an unscrambled Sobol point set through the standard-normal
    quantile function. The result is deterministic. The implementation
    skips the Sobol origin, clamps quantile inputs away from 0 and 1,
    and rounds ``count`` up to a power of two.

    Parameters
    ----------
    dimension : int
        Number of coordinates per point. Zero yields an empty
        ``(n, 0)`` tensor, which is what a deterministic map consumes.
    count : int
        Requested point count; rounded up to a power of two.
    device : torch.device
        Device to place the result on.
    dtype : torch.dtype
        Floating dtype of the result.

    Returns
    -------
    torch.Tensor
        Shape ``(n, dimension)``.
    """
    n = _next_power_of_two(count)
    if dimension == 0:
        return torch.zeros(n, 0, device=device, dtype=dtype)
    engine = torch.quasirandom.SobolEngine(dimension=dimension, scramble=False)
    unit = engine.draw(n + 1, dtype=torch.float64)[1:]
    unit = unit.clamp(min=_QUANTILE_EPS, max=1.0 - _QUANTILE_EPS)
    return torch.special.ndtri(unit).to(device=device, dtype=dtype)


def _reduce_to_batch(log_prob: torch.Tensor) -> torch.Tensor:
    """Sum a factor's log-density down to one number per row.

    A kernel applied along a sequence returns one density per position
    (``(batch, seq)``) where the same kernel on a single position
    returns ``(batch,)``. The positions are independent factors of the
    same row's density, so the row's contribution is their sum; leaving
    the axis in place would let it broadcast against the rest of the
    chain and count every other factor once per position.
    """
    if log_prob.dim() <= 1:
        return log_prob
    return log_prob.reshape(log_prob.shape[0], -1).sum(dim=-1)


def _fold_features(value: torch.Tensor) -> torch.Tensor:
    """Fold a chain intermediate to one feature axis per row.

    A kernel applied along a sequence carries a position axis its
    declared codomain does not, and the value the next factor expects
    carries only the declared one. Scoring the two unfolded broadcasts
    a ``(batch, seq, d)`` mean against a ``(batch, d)`` value into a
    finite number that is the density of nothing, so the position axis
    is folded into the feature axis and the next factor reads it back
    from the width it declares.
    """
    if value.dim() <= 2:
        return value
    return value.reshape(value.shape[0], -1)


def chain_dimensions(
    factors: "collections.abc.Sequence[ContinuousMorphism]", x: torch.Tensor
) -> list[int] | None:
    """Per-factor base-coordinate counts along a chain, or None.

    A factor's count can depend on the shape of what reaches it, so
    the chain is walked once with the coordinates held at zero. That
    pushes each kernel's median forward, which costs a forward pass
    and settles the shapes without consuming a point set the caller
    has not built yet.
    """
    dimensions: list[int] = []
    probe = dimension_probe(x)
    dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    for factor in factors:
        dimension = factor.base_dimension(probe)
        if dimension is None:
            return None
        dimensions.append(dimension)
        zeros = torch.zeros(probe.shape[0], dimension, device=probe.device, dtype=dtype)
        probe = factor.push_base(probe, zeros)
    return dimensions


def chain_push_base(
    factors: "collections.abc.Sequence[ContinuousMorphism]",
    x: torch.Tensor,
    base: torch.Tensor,
    dimensions: list[int],
) -> torch.Tensor:
    """Thread base coordinates through a chain, one block per factor.

    Each factor consumes its own contiguous block, so no two factors
    read the same coordinate and the composite map is the pushforward
    of a single point set through the whole chain rather than
    per-factor rules glued together by index. Sharing coordinates
    across factors would resolve some directions of the joint twice
    and leave others unexplored.
    """
    offset = 0
    current = x
    for factor, dimension in zip(factors, dimensions):
        current = factor.push_base(current, base[:, offset : offset + dimension])
        offset += dimension
    return current


def chain_marginal_quadrature(
    factors: "collections.abc.Sequence[ContinuousMorphism]",
    x: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """A deterministic rule for the law a whole chain induces on its end.

    Push one point set of dimension ``sum(chain_dimensions(...))``
    through every factor. The rule returns ``n`` terminal nodes without
    multiplying the node count at each link.

    Parameters
    ----------
    factors : Sequence[ContinuousMorphism]
        The chain, in application order. A single-element sequence
        defers to that morphism's own rule.
    x : torch.Tensor
        Conditioning inputs. Shape ``(batch, *domain)``.
    count : int
        Requested node count; rounded up to a power of two.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or None
        Nodes of shape ``(n, batch, *event)`` and log-weights of
        shape ``(n,)``, or ``None`` when any factor has no
        reparameterization.
    """
    if not factors:
        raise ValueError(
            "chain_marginal_quadrature: an empty chain induces no law; "
            "pass at least one factor."
        )
    if len(factors) == 1:
        return factors[0].marginal_quadrature(x, count)
    dimensions = chain_dimensions(factors, x)
    if dimensions is None:
        return None
    total = sum(dimensions)
    batch = x.shape[0]
    dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
    base = sobol_normal_points(total, count, x.device, dtype)
    n = base.shape[0]
    x_rows = x.unsqueeze(0).expand(n, *x.shape).reshape(n * batch, *x.shape[1:])
    base_rows = base.unsqueeze(1).expand(n, batch, total).reshape(n * batch, total)
    pushed = chain_push_base(factors, x_rows, base_rows, dimensions)
    nodes = pushed.reshape(n, batch, *pushed.shape[1:])
    log_weights = torch.full(
        (n,), -math.log(float(n)), device=nodes.device, dtype=nodes.dtype
    )
    return nodes, log_weights


class ContinuousMorphism(nn.Module, ABC):
    """Abstract base for morphisms involving continuous spaces.

    Subclasses must implement ``log_prob`` and ``rsample``. The
    composition operator ``>>`` and product operator ``@`` are
    provided and dispatch to SampledComposition and
    ProductContinuousMorphism respectively.

    Unlike discrete Morphism (which materializes a full tensor),
    ContinuousMorphism is defined operationally: it can evaluate
    log-densities and generate reparameterized samples.

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        Source space.
    codomain : SetObject or ContinuousSpace
        Target space.
    """

    def __init__(self, domain: AnySpace, codomain: AnySpace) -> None:
        super().__init__()
        self._domain = domain
        self._codomain = codomain

    @property
    def domain(self) -> AnySpace:
        """Source space."""
        return self._domain

    @property
    def codomain(self) -> AnySpace:
        """Target space."""
        return self._codomain

    @property
    def support(self) -> _constraints.Constraint:
        """The support constraint of the distribution this morphism samples
        from, in the form of a `torch.distributions.constraints.Constraint`.

        Used by variational guides ([`quivers.inference.AutoNormalGuide`][quivers.inference.AutoNormalGuide],
        [`quivers.inference.AutoDeltaGuide`][quivers.inference.AutoDeltaGuide]) to determine the
        correct bijector that maps an unconstrained variational
        approximation back to the constrained support of the prior, so
        that samples used to evaluate the prior's ``log_prob`` lie inside
        its support (avoiding ``Expected value to be within the support
        of the distribution`` errors).

        Subclasses representing a constrained distribution family
        (``HalfNormal``, ``Beta``, ``Uniform``, ``Dirichlet``,
        ``LogitNormal``, ``Wishart``, …) should override this property
        to return the appropriate constraint. The default is
        `torch.distributions.constraints.real`, which is correct
        for unconstrained families like ``Normal`` and discrete
        codomains (where the guide skips the site anyway).
        """
        return _constraints.real

    @abstractmethod
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability (density) of y given x.

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape (batch,) for discrete domain or
            (batch, domain_dim) for continuous domain.
        y : torch.Tensor
            Outputs. Shape (batch,) for discrete codomain or
            (batch, codomain_dim) for continuous codomain.

        Returns
        -------
        torch.Tensor
            Log-probabilities/densities. Shape (batch,).
        """
        ...

    @abstractmethod
    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Reparameterized samples from p(. | x).

        Gradients flow through the returned samples back to the
        parameters of this morphism (and to x if the domain is
        continuous).

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape (batch,) or (batch, domain_dim).
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples. Shape (*sample_shape, batch, codomain_dim) for
            continuous codomain, or (*sample_shape, batch) for discrete.
        """
        ...

    def has_conditional_density(self) -> bool:
        """Whether
        [`log_prob`][quivers.continuous.morphisms.ContinuousMorphism.log_prob]
        evaluates a density rather than raising.

        Every kernel whose conditional law is a named family answers
        yes. A kernel that denotes a *program* answers no: its density
        at a value marginalizes the program's internal draws, which no
        closed form covers, and its `log_prob` says so by raising.

        A caller deciding between two constructions needs that answer
        before it calls, not as an exception afterwards, so the
        capability is declared rather than discovered. The parallel
        with
        [`point_mass_value`][quivers.continuous.morphisms.ContinuousMorphism.point_mass_value]
        and
        [`base_dimension`][quivers.continuous.morphisms.ContinuousMorphism.base_dimension]
        is exact: each reports a structural property of the kernel that
        determines which exact treatment is open to a caller, and none
        of them is a probe by trial.

        Returns
        -------
        bool
            True for a kernel with an evaluable conditional density.
        """
        return True

    def point_mass_value(self, x: torch.Tensor) -> torch.Tensor | None:
        """The single value this kernel puts all of its mass on, or None.

        A morphism whose conditional law is a Dirac delta
        :math:`\\delta_{T(x)}` returns :math:`T(x)`; every other
        morphism returns ``None``. The distinction is what lets
        [`SampledComposition`][quivers.continuous.morphisms.SampledComposition]
        collapse an integral over a degenerate intermediate to a
        single evaluation, which is exact rather than approximate.

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape ``(batch,)`` or ``(batch, domain_dim)``.

        Returns
        -------
        torch.Tensor or None
            The deterministic image of ``x``, or ``None`` when the
            kernel is genuinely stochastic.
        """
        del x
        return None

    def base_dimension(self, x: torch.Tensor) -> int | None:
        """Standard-normal coordinates this kernel's reparameterization reads.

        A morphism that can be written :math:`y = T_x(\\varepsilon)`
        with :math:`\\varepsilon` standard normal reports how many
        coordinates :math:`T_x` consumes at this input; every other
        morphism reports ``None``. A deterministic map consumes none
        and reports ``0``.

        The count may depend on ``x``: an embedding kernel reading a
        ``(batch, seq)`` index matrix places one Gaussian per position,
        so it consumes ``seq * dim`` coordinates where the same kernel
        on a ``(batch,)`` index vector consumes ``dim``.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning inputs. Shape ``(batch,)`` or
            ``(batch, domain_dim)``.

        Returns
        -------
        int or None
            The coordinate count, or ``None`` when this morphism has
            no reparameterization to offer.
        """
        del x
        if type(self).point_mass_value is not ContinuousMorphism.point_mass_value:
            return 0
        return None

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Push standard-normal coordinates through the reparameterization.

        Evaluates :math:`T_x(\\varepsilon)` for the map
        [`base_dimension`][quivers.continuous.morphisms.ContinuousMorphism.base_dimension]
        describes. The map is a pure function of ``(x, base)``: it
        reads no random state, which is what lets a caller build a
        quadrature out of it and get the same nodes on every call.

        The default covers the degenerate case, where the map ignores
        its (empty) coordinates and returns the point mass.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning inputs. Shape ``(batch, *domain)``.
        base : torch.Tensor
            Standard-normal coordinates. Shape ``(batch, dimension)``
            for the dimension `base_dimension` reports at ``x``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, *event)``.
        """
        del base
        value = self.point_mass_value(x)
        if value is None:
            raise ValueError(
                f"{type(self).__name__}.push_base: this morphism "
                f"declares no reparameterization, so there is nothing "
                f"to push coordinates through. Override "
                f"`base_dimension` and `push_base` together, or leave "
                f"both at their defaults so callers see the absence "
                f"rather than a wrong value."
            )
        return value

    def push_base_scored(
        self, x: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Push coordinates through the reparameterization and score the draw.

        Returns the value
        [`push_base`][quivers.continuous.morphisms.ContinuousMorphism.push_base]
        computes together with the log-density of the draw those
        coordinates select. For most kernels the draw is the value, so
        the density is ``log_prob(x, value)``; a kernel whose draw
        carries more than its value, as a scan's trajectory carries
        every state before the final one, overrides this to score the
        whole draw.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning inputs. Shape ``(batch, *domain)``.
        base : torch.Tensor
            Standard-normal coordinates. Shape ``(batch, dimension)``
            for the dimension `base_dimension` reports at ``x``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The value, shape ``(batch, *event)``, and the draw's
            log-density, shape ``(batch,)``.

        Raises
        ------
        ValueError
            If the morphism declares no reparameterization.
        """
        value = self.push_base(x, base)
        return value, _reduce_to_batch(self.log_prob(x, value))

    def marginal_quadrature(
        self, x: torch.Tensor, count: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """A deterministic rule for integrating against ``p(. | x)``.

        Returns ``(nodes, log_weights)`` approximating

        .. math::

            \\int p(y \\mid x) \\, \\varphi(y) \\, dy
            \\;\\approx\\;
            \\sum_i \\exp(\\log w_i) \\, \\varphi(y_i)

        Point masses return one unit-weight node. Reparameterized
        kernels return equally weighted Sobol nodes produced by
        ``push_base``. Kernels without either representation return
        ``None``. This method does not consume random state.

        Parameters
        ----------
        x : torch.Tensor
            Conditioning inputs. Shape ``(batch,)`` or
            ``(batch, domain_dim)``.
        count : int
            Requested number of nodes. An implementation may return
            fewer (an exact rule needs one) or round up to the count
            its construction is balanced at.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] or None
            Nodes and log-weights, or ``None`` when this morphism
            provides no deterministic rule.
        """
        value = self.point_mass_value(x)
        if value is not None:
            nodes = value.unsqueeze(0)
            log_weights = torch.zeros(1, device=nodes.device, dtype=nodes.dtype)
            return nodes, log_weights
        dimension = self.base_dimension(x)
        if dimension is None:
            return None
        batch = x.shape[0]
        dtype = x.dtype if x.is_floating_point() else torch.get_default_dtype()
        base = sobol_normal_points(dimension, count, x.device, dtype)
        n = base.shape[0]
        x_rows = x.unsqueeze(0).expand(n, *x.shape).reshape(n * batch, *x.shape[1:])
        base_rows = (
            base.unsqueeze(1).expand(n, batch, dimension).reshape(n * batch, dimension)
        )
        pushed = self.push_base(x_rows, base_rows)
        nodes = pushed.reshape(n, batch, *pushed.shape[1:])
        log_weights = torch.full(
            (n,), -math.log(float(n)), device=nodes.device, dtype=nodes.dtype
        )
        return nodes, log_weights

    def sample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Non-reparameterized samples (no gradient through samples).

        Parameters
        ----------
        x : torch.Tensor
            Inputs.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples (detached from computation graph).
        """
        with torch.no_grad():
            return self.rsample(x, sample_shape)

    def __rshift__(self, other: object) -> ContinuousMorphism:
        """Composition via ancestral sampling: self >> other."""
        if isinstance(other, ContinuousMorphism):
            return SampledComposition(self, other)

        if isinstance(other, Morphism):
            return SampledComposition(self, DiscreteAsContinuous(other))
        return NotImplemented

    def __rrshift__(self, other: object) -> ContinuousMorphism:
        """Handle discrete_morphism >> continuous_morphism."""

        if isinstance(other, Morphism):
            return SampledComposition(DiscreteAsContinuous(other), self)
        return NotImplemented

    def __matmul__(self, other: object) -> ProductContinuousMorphism:
        """Independent product: self @ other."""
        if isinstance(other, ContinuousMorphism):
            return ProductContinuousMorphism(self, other)

        if isinstance(other, Morphism):
            return ProductContinuousMorphism(self, DiscreteAsContinuous(other))
        return NotImplemented

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.domain!r} -> {self.codomain!r})"


class MarginalizedFactor(ContinuousMorphism):
    """Score-suppressed wrapper for a marginalized block's live sites.

    An ungrouped ``marginalize`` block keeps its latent draw and its
    terminal observe as live sites so a forward trace still produces
    the sampled coordinate and response (ancestral sampling and
    synthetic-data generation both read those sites). Their densities,
    however, are carried once by the block's integrated score step, so
    adding them to the joint again would double-count the very factors
    the marginal already integrates. This wrapper delegates sampling to
    the base morphism yet reports a zero log-density, keeping the joint
    free of the raw per-draw factor while preserving forward behaviour.

    Parameters
    ----------
    base : ContinuousMorphism
        The underlying family whose sampling behaviour is preserved.
    """

    def __init__(self, base: ContinuousMorphism) -> None:
        super().__init__(base.domain, base.codomain)
        self.base = base

    @property
    def support(self) -> _constraints.Constraint:
        """Delegate the support constraint to the wrapped family."""
        return self.base.support

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the base family (forward behaviour is preserved)."""
        return self.base.rsample(x, sample_shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Report a zero log-density.

        The factor's density is carried by the block's integrated score
        step; contributing it here would double-count it in the joint.
        """
        del x
        return torch.zeros((), device=y.device, dtype=torch.get_default_dtype())


class SampledComposition(ContinuousMorphism):
    """Composition of morphisms via ancestral sampling.

    ``rsample`` draws from ``left`` and then ``right``. ``log_prob``
    sums over a discrete intermediate. For a stochastic continuous
    intermediate it scores the deterministic reference path, not the
    endpoint marginal.

    Parameters
    ----------
    left : ContinuousMorphism
        First morphism (applied first).
    right : ContinuousMorphism
        Second morphism (applied second).
    n_intermediate : int
        Node count for the deterministic rule this composition offers
        through
        [`marginal_quadrature`][quivers.continuous.morphisms.ContinuousMorphism.marginal_quadrature]
        to a caller that asks for one. The composite density does not
        ask: it scores the canonical path and integrates nothing, so
        this count does not reach `log_prob`.
    """

    def __init__(
        self,
        left: ContinuousMorphism,
        right: ContinuousMorphism,
        n_intermediate: int = 100,
    ) -> None:
        super().__init__(left.domain, right.codomain)
        self.left = left
        self.right = right
        self.n_intermediate = n_intermediate

    @property
    def factors(self) -> tuple[ContinuousMorphism, ...]:
        """The composition flattened into its non-composite factors.

        ``(a >> b) >> c`` and ``a >> (b >> c)`` both report
        ``(a, b, c)``: association is invisible to the kernel the
        composition denotes, and every intermediate between adjacent
        factors is an object the chain integrates over. A caller that
        wants those intermediates as named sites walks this tuple.
        """
        chain: list[ContinuousMorphism] = []
        for side in (self.left, self.right):
            if isinstance(side, SampledComposition):
                chain.extend(side.factors)
            else:
                chain.append(side)
        return tuple(chain)

    def base_dimension(self, x: torch.Tensor) -> int | None:
        """Total coordinates the whole chain's reparameterization reads.

        A chain is reparameterized by reparameterizing each factor and
        threading the result forward, so its coordinate budget is the
        sum of its factors'. One factor without a reparameterization
        leaves the chain without one.
        """
        dimensions = chain_dimensions(self.factors, x)
        if dimensions is None:
            return None
        return sum(dimensions)

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Thread the coordinates through the chain, factor by factor.

        Each factor consumes its own contiguous block of ``base``, so
        no two factors share a coordinate.
        """
        dimensions = chain_dimensions(self.factors, x)
        if dimensions is None:
            raise ValueError(
                f"SampledComposition.push_base: a factor of this chain "
                f"declares no reparameterization, so the chain has "
                f"none either. The factors are "
                f"{[type(f).__name__ for f in self.factors]!r}."
            )
        return chain_push_base(self.factors, x, base, dimensions)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Ancestral sampling: y ~ f(x, .), then z ~ g(y, .).

        Parameters
        ----------
        x : torch.Tensor
            Inputs to the composition.
        sample_shape : torch.Size
            Additional sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples from the composed morphism.
        """
        y = self.left.rsample(x, sample_shape)
        if len(sample_shape) > 0:
            leading = y.shape[: len(sample_shape)]
            batch = x.shape[0]
            flat_size = int(torch.tensor(leading).prod().item()) * batch
            if y.dim() > len(sample_shape) + 1:
                event_dims = y.shape[len(sample_shape) + 1 :]
                flat_y = y.reshape(flat_size, *event_dims)
            else:
                flat_y = y.reshape(flat_size)
        else:
            flat_y = y
        z = self.right.rsample(flat_y)
        if len(sample_shape) > 0:
            batch = x.shape[0]
            if z.dim() > 1:
                event_dims = z.shape[1:]
                z = z.reshape(*sample_shape, batch, *event_dims)
            else:
                z = z.reshape(*sample_shape, batch)
        return z

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability of y given x through the composition.

        A discrete intermediate is marginalized exactly, by finite
        summation over its elements. A continuous one is not
        marginalized at all: the chain is scored along the canonical
        path
        `_log_prob_reference_path`
        describes, which is exact where an integral would have been
        approximate, and is a pure function of ``(x, y)`` where a rule
        would have made it a function of the node count as well.

        Both branches return the same number when every intermediate
        is degenerate, which is the case the two readings share.

        Parameters
        ----------
        x : torch.Tensor
            Inputs. Shape (batch,) or (batch, dom_dim).
        y : torch.Tensor
            Outputs. Shape (batch,) or (batch, cod_dim).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        intermediate = self.left.codomain
        if isinstance(intermediate, SetObject):
            return self._log_prob_exact(x, y, intermediate)
        else:
            return self._log_prob_reference_path(x, y)

    def _log_prob_exact(
        self, x: torch.Tensor, z: torch.Tensor, intermediate: SetObject
    ) -> torch.Tensor:
        """Exact log-prob via finite summation over discrete intermediate."""
        batch = x.shape[0]
        n_y = intermediate.size
        all_y = torch.arange(n_y, device=x.device)
        x.unsqueeze(1).expand(
            batch if x.dim() == 1 else x.shape[0],
            n_y,
            *(() if x.dim() == 1 else x.shape[1:]),
        )
        if x.dim() == 1:
            x_flat = x.unsqueeze(1).expand(batch, n_y).reshape(-1)
        else:
            x_flat = (
                x.unsqueeze(1).expand(batch, n_y, x.shape[-1]).reshape(-1, x.shape[-1])
            )
        y_flat = all_y.unsqueeze(0).expand(batch, n_y).reshape(-1)
        log_f = self.left.log_prob(x_flat, y_flat).reshape(batch, n_y)
        if z.dim() == 1:
            z_flat = z.unsqueeze(1).expand(batch, n_y).reshape(-1)
        else:
            z_flat = (
                z.unsqueeze(1).expand(batch, n_y, z.shape[-1]).reshape(-1, z.shape[-1])
            )
        log_g = self.right.log_prob(y_flat, z_flat).reshape(batch, n_y)
        return torch.logsumexp(log_f + log_g, dim=1)

    def _log_prob_reference_path(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Score every factor along the chain's canonical path.

        For a stochastic continuous intermediate, this method does not
        evaluate the endpoint marginal. It binds each prefix
        intermediate to ``push_base`` at zero coordinates and scores
        every factor once:

        .. math::

            \\sum_{k<n} \\log p_k(y_k \\mid y_{k-1})
            + \\log p_n(z \\mid y_{n-1}),
            \\qquad y_k = T_{y_{k-1}}(0).

        The result is deterministic and represents the path joint, not
        the marginal density of ``z``. Degenerate factors pass their
        point mass forward without adding a density term.

        Parameters
        ----------
        x : torch.Tensor
            Inputs to the chain. Shape ``(batch, *domain)``.
        z : torch.Tensor
            Value at the chain's codomain. Shape ``(batch, *event)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)``.

        Raises
        ------
        ValueError
            If a stochastic prefix factor cannot provide or score a
            reference intermediate.
        """
        factors = self.factors
        prefix = factors[:-1]
        last = factors[-1]
        total: torch.Tensor | None = None
        current = x
        for factor in prefix:
            degenerate = factor.point_mass_value(current)
            if degenerate is not None:
                current = degenerate
                continue
            dimension = factor.base_dimension(current)
            if dimension is None:
                raise ValueError(
                    f"SampledComposition.log_prob: the intermediate "
                    f"object {factor.codomain!r} is continuous and "
                    f"{type(factor).__name__} declares no "
                    f"reparameterization, so the chain has no canonical "
                    f"value to bind it to. Bind the intermediate to a "
                    f"draw step of its own and score the chain factor "
                    f"by factor, or give the factor a `base_dimension` "
                    f"/ `push_base` pair."
                )
            if not factor.has_conditional_density():
                raise ValueError(
                    f"SampledComposition.log_prob: "
                    f"{type(factor).__name__} has no conditional "
                    f"density, so its step of the chain cannot be "
                    f"scored. Bind the intermediate to a draw step of "
                    f"its own and score the factor's own draws."
                )
            dtype = (
                current.dtype
                if current.is_floating_point()
                else torch.get_default_dtype()
            )
            base = torch.zeros(
                current.shape[0], dimension, device=current.device, dtype=dtype
            )
            following, term = factor.push_base_scored(current, base)
            total = term if total is None else total + term
            current = following
        term = _reduce_to_batch(last.log_prob(_fold_features(current), z))
        return term if total is None else total + term


class ProductContinuousMorphism(ContinuousMorphism):
    """Independent product of two continuous morphisms.

    Given f: A -> B and g: C -> D, produces f @ g: (A, C) -> (B, D)
    where p_{f@g}((y,z) | (x,w)) = f(y | x) * g(z | w).

    Domain inputs are concatenated: (x, w) as a single vector.
    Codomain outputs are concatenated: (y, z) as a single vector.
    For discrete components, indices are embedded as 1-d floats.

    Parameters
    ----------
    left : ContinuousMorphism
        Left factor morphism.
    right : ContinuousMorphism
        Right factor morphism.
    """

    def __init__(self, left: ContinuousMorphism, right: ContinuousMorphism) -> None:
        dom = _combine_spaces(left.domain, right.domain)
        cod = _combine_spaces(left.codomain, right.codomain)
        super().__init__(dom, cod)
        self.left = left
        self.right = right
        self._left_dom_dim = _event_dim(left.domain)
        self._right_dom_dim = _event_dim(right.domain)
        self._left_cod_dim = _event_dim(left.codomain)
        self._right_cod_dim = _event_dim(right.codomain)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        x_left, x_right = self._split_input(x)
        y_left = self.left.rsample(x_left, sample_shape)
        y_right = self.right.rsample(x_right, sample_shape)
        if y_left.dim() < y_right.dim():
            y_left = y_left.unsqueeze(-1)
        elif y_right.dim() < y_left.dim():
            y_right = y_right.unsqueeze(-1)
        return torch.cat([y_left, y_right], dim=-1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_left, x_right = self._split_input(x)
        y_left = y[..., : self._left_cod_dim]
        y_right = y[..., self._left_cod_dim :]
        if _is_discrete(self.left.codomain):
            y_left = y_left.squeeze(-1).long()
        if _is_discrete(self.right.codomain):
            y_right = y_right.squeeze(-1).long()
        return self.left.log_prob(x_left, y_left) + self.right.log_prob(
            x_right, y_right
        )

    def base_dimension(self, x: torch.Tensor) -> int | None:
        """Sum of the two factors' coordinate budgets at their own inputs."""
        x_left, x_right = self._split_input(dimension_probe(x))
        left = self.left.base_dimension(x_left)
        right = self.right.base_dimension(x_right)
        if left is None or right is None:
            return None
        return left + right

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Push each factor's own coordinate block through that factor.

        The factors are independent given the input, so the product's
        reparameterization is the pair of theirs on disjoint
        coordinate blocks, concatenated along the feature axis exactly
        as `rsample` concatenates its draws.
        """
        x_left, x_right = self._split_input(x)
        probe_left, probe_right = self._split_input(dimension_probe(x))
        left_dimension = self.left.base_dimension(probe_left)
        right_dimension = self.right.base_dimension(probe_right)
        if left_dimension is None or right_dimension is None:
            raise ValueError(
                f"ProductContinuousMorphism.push_base: factor "
                f"{type(self.left).__name__} @ "
                f"{type(self.right).__name__} declares no "
                f"reparameterization, so the product has none either."
            )
        y_left = self.left.push_base(x_left, base[:, :left_dimension])
        y_right = self.right.push_base(x_right, base[:, left_dimension:])
        if y_left.dim() < y_right.dim():
            y_left = y_left.unsqueeze(-1)
        elif y_right.dim() < y_left.dim():
            y_right = y_right.unsqueeze(-1)
        return torch.cat([y_left, y_right], dim=-1)

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split concatenated domain input into left and right parts."""
        d = self._left_dom_dim
        x_left = x[..., :d]
        x_right = x[..., d:]
        if _is_discrete(self.left.domain) and x_left.dim() > 1:
            x_left = x_left.squeeze(-1).long()
        elif _is_discrete(self.left.domain):
            x_left = x_left.long()
        if _is_discrete(self.right.domain) and x_right.dim() > 1:
            x_right = x_right.squeeze(-1).long()
        elif _is_discrete(self.right.domain):
            x_right = x_right.long()
        return (x_left, x_right)


class FanOutMorphism(ContinuousMorphism):
    """Fan-out morphism: copy input to N morphisms, concatenate outputs.

    Given f_1: A -> B_1, f_2: A -> B_2, ..., f_N: A -> B_N,
    produces fan(f_1, ..., f_N): A -> B_1 * B_2 * ... * B_N
    where the input A is copied to all N morphisms.

    Unlike the tensor product (f @ g), which takes a product domain
    (A * C), fan-out feeds the same input to all morphisms. This
    implements the diagonal morphism Delta: A -> A^N followed by
    the product f_1 @ f_2 @ ... @ f_N.

    Parameters
    ----------
    components : list[ContinuousMorphism]
        The morphisms to fan out to. All must share the same domain.
    """

    def __init__(self, components: list) -> None:

        if not components:
            raise ValueError("fan-out requires at least one component")
        # Backend-agnostic V-Cat morphisms (those that aren't
        # already ContinuousMorphism subclasses) get wrapped in a
        # deterministic continuous adapter so the FanOut's rsample
        # / log_prob loop can dispatch uniformly. The wrapping
        # exposes the V-Cat tensor through a categorical
        # ``rsample`` that gathers / contracts the tensor against
        # the input; ``log_prob`` evaluates the V-Cat tensor as a
        # categorical likelihood when meaningful.
        wrapped_components: list[ContinuousMorphism] = []
        for c in components:
            if isinstance(c, ContinuousMorphism):
                wrapped_components.append(c)
            elif isinstance(c, _CatMorphism):
                wrapped_components.append(DiscreteAsContinuous(c))
            else:
                raise TypeError(
                    f"fan-out: component of type "
                    f"{type(c).__name__} is neither a "
                    f"ContinuousMorphism nor a V-Cat Morphism"
                )
        domain = wrapped_components[0].domain
        for i, c in enumerate(wrapped_components[1:], 1):
            dom_dim = _event_dim(domain)
            c_dim = _event_dim(c.domain)
            if dom_dim != c_dim:
                raise TypeError(
                    f"fan-out: component {i} domain dim {c_dim} != component 0 domain dim {dom_dim}"
                )
        codomain = wrapped_components[0].codomain
        for c in wrapped_components[1:]:
            codomain = _combine_spaces(codomain, c.codomain)
        super().__init__(domain, codomain)
        self._components = torch.nn.ModuleList(wrapped_components)
        self._cod_dims = [_event_dim(c.codomain) for c in wrapped_components]

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from all components and concatenate outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (broadcast to all components).
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Concatenated outputs from all components.
        """
        outs = []
        for comp in self._components:
            y = cast(ContinuousMorphism, comp).rsample(x, sample_shape)
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            outs.append(y)
        return torch.cat(outs, dim=-1)

    def _component_dimensions(self, x: torch.Tensor) -> list[int] | None:
        """Each component's coordinate budget at the shared input."""
        dimensions: list[int] = []
        probe = dimension_probe(x)
        for comp in self._components:
            dimension = cast(ContinuousMorphism, comp).base_dimension(probe)
            if dimension is None:
                return None
            dimensions.append(dimension)
        return dimensions

    def base_dimension(self, x: torch.Tensor) -> int | None:
        """Sum of the components' coordinate budgets.

        Fan-out copies its input to independent components, so their
        reparameterizations share the input and nothing else.
        """
        dimensions = self._component_dimensions(x)
        if dimensions is None:
            return None
        return sum(dimensions)

    def push_base(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """Push each component's own coordinate block through it.

        The blocks are disjoint and the outputs concatenate along the
        feature axis, matching the layout `rsample` and `log_prob`
        already use for the fan's codomain.
        """
        dimensions = self._component_dimensions(x)
        if dimensions is None:
            raise ValueError(
                "FanOutMorphism.push_base: component(s) "
                f"{[type(c).__name__ for c in self._components]!r} "
                "declare no reparameterization, so the fan has none "
                "either."
            )
        outs = []
        offset = 0
        for comp, dimension in zip(self._components, dimensions):
            y = cast(ContinuousMorphism, comp).push_base(
                x, base[:, offset : offset + dimension]
            )
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            outs.append(y)
            offset += dimension
        return torch.cat(outs, dim=-1)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability: sum of component log-probs.

        Parameters
        ----------
        x : torch.Tensor
            Input (same for all components).
        y : torch.Tensor
            Concatenated output values.

        Returns
        -------
        torch.Tensor
            Sum of log-probabilities. Shape ``(batch,)``.
        """
        lp = torch.zeros(x.shape[0], device=x.device)
        offset = 0
        for comp_mod, d in zip(self._components, self._cod_dims):
            comp = cast(ContinuousMorphism, comp_mod)
            y_slice = y[..., offset : offset + d]
            if _is_discrete(comp.codomain):
                y_slice = y_slice.squeeze(-1).long()
            lp = lp + comp.log_prob(x, y_slice)
            offset += d
        return lp


class DiscreteAsContinuous(ContinuousMorphism):
    """Wrap a discrete Morphism as a ContinuousMorphism.

    Enables composition between discrete and continuous morphisms
    via the >> operator. The kernel is the Markov kernel the wrapped
    morphism's nonnegative matrix induces: row ``x`` normalized to sum
    to one, ``P(y | x) = t[x, y] / sum_y' t[x, y']``. Sampling and
    scoring both use these normalized rows, so the density `log_prob`
    reports is the law `rsample` draws from; a stochastic matrix is
    its own normalization.

    Note
    ----
    Sampling from a discrete distribution is NOT reparameterizable.
    Gradients do not flow through the discrete samples back to the
    left morphism's parameters. Use score function estimators
    (REINFORCE) if gradients through discrete choices are needed.

    Parameters
    ----------
    inner : Morphism
        The discrete morphism to wrap.
    """

    def __init__(self, inner: object) -> None:

        if not isinstance(inner, Morphism):
            raise TypeError(f"expected a discrete Morphism, got {type(inner).__name__}")
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner
        self._inner_module = inner.module()

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability from the discrete tensor.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).
        y : torch.Tensor
            Codomain indices. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape (batch,).
        """
        rows = self._rows(x)
        return torch.log(rows.gather(-1, y.long().unsqueeze(-1)).squeeze(-1))

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        """The normalized rows of the wrapped matrix at ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).

        Returns
        -------
        torch.Tensor
            Shape ``(batch, |codomain|)``, each row a probability
            vector.

        Raises
        ------
        ValueError
            If a row has a negative entry or sums to zero, so it
            induces no distribution.
        """
        rows = self._inner.tensor[x.long()]
        if bool((rows < 0).any()):
            raise ValueError(
                "DiscreteAsContinuous: the wrapped matrix has a negative "
                "entry, so its rows are not distributions"
            )
        totals = rows.sum(dim=-1, keepdim=True)
        if bool((totals <= 0).any()):
            raise ValueError(
                "DiscreteAsContinuous: a row of the wrapped matrix sums to "
                "zero, so it induces no distribution over the codomain"
            )
        return rows / totals

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the categorical distribution defined by the tensor.

        Note: not reparameterizable. Gradients do not flow through
        the returned samples.

        Parameters
        ----------
        x : torch.Tensor
            Domain indices. Shape (batch,).
        sample_shape : torch.Size
            Additional sample dimensions.

        Returns
        -------
        torch.Tensor
            Sampled codomain indices. Shape (*sample_shape, batch).
        """
        probs = self._rows(x)
        n_samples = (
            int(torch.Size(sample_shape).numel()) if len(sample_shape) > 0 else 1
        )
        samples = torch.multinomial(probs, n_samples, replacement=True)
        if len(sample_shape) == 0:
            return samples.squeeze(-1)
        else:
            return samples.T.reshape(*sample_shape, -1)


def _event_dim(space: AnySpace) -> int:
    """Get the event dimensionality of a space.

    Discrete spaces are treated as 1-dimensional (index encoded as float).
    """
    if isinstance(space, ContinuousSpace):
        return space.dim
    return 1


def _combine_spaces(a: AnySpace, b: AnySpace) -> AnySpace:
    """Create a product of two spaces (possibly mixed types).

    For two continuous spaces, returns a ProductSpace.
    For mixed types, wraps discrete spaces as Euclidean(1).
    """

    def _as_continuous(s: AnySpace) -> ContinuousSpace:
        if isinstance(s, ContinuousSpace):
            return s
        return Euclidean(name=f"idx({s!r})", dim=1)

    return ProductSpace(components=(_as_continuous(a), _as_continuous(b)))


__all__ = [
    "AnySpace",
    "ContinuousMorphism",
    "DiscreteAsContinuous",
    "FanOutMorphism",
    "MarginalizedFactor",
    "ProductContinuousMorphism",
    "SampledComposition",
]
