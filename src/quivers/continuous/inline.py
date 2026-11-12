"""Inline distribution morphisms for monadic programs.

These ContinuousMorphisms parameterize distributions using either
fixed literal values or direct variable values, without learned
neural-network transformations. Used by the DSL compiler for inline
draw steps like::

    draw x ~ LogitNormal(0.0, 1.0)                 # all fixed
    draw b ~ Bernoulli(x)                           # direct variable
    draw r ~ TruncatedNormal(mu, sigma, 0.0, 1.0)  # mixed

Terminology
-----------
- **Fixed**: all distribution parameters are float literals known at
  compile time. The morphism has domain ``Unit`` (terminal object)
  and ignores its input.
- **Direct**: some or all parameters come from bound variables at
  runtime. The morphism uses the input tensor directly as
  distribution parameters (no learned neural-net transformation).

Architecture
------------
Each family is described once by a `FamilySpec` in
`quivers.continuous.family_spec`. The factories and builders
below dispatch on the registered spec rather than maintaining
per-family hard-coded functions; specials (truncated, Dirichlet,
uniform, transformed) override the generic dispatch by setting
``fixed_factory_override`` / ``mixed_builder_override`` on their
FamilySpec entry.
"""

from __future__ import annotations

from typing import Literal
import math
from collections.abc import Callable, Mapping
import torch
import torch.distributions as D
from torch.distributions import constraints as _constraints
from quivers.core.objects import Unit
from quivers.continuous._lkj import OnionLKJCholesky
from quivers.continuous._matrix import InverseWishart, MatrixNormal
from quivers.continuous.log_space import LogGamma, LogitBeta
from quivers.continuous._ordered import OrderedLogistic, OrderedProbit
from quivers.continuous._zip_hurdle import (
    HurdlePoisson,
    MixtureNormal,
    ZeroInflatedPoisson,
    ZeroOneInflatedBeta,
)
from quivers.continuous.measure import (
    Independent as _MeasureIndependent,
    Measure as _Measure,
    Mixture as _MeasureMixture,
    Normalize as _MeasureNormalize,
    PointMass as _MeasurePointMass,
    Pushforward as _MeasurePushforward,
    Restrict as _MeasureRestrict,
)
from quivers.continuous.spaces import Euclidean
from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.continuous.family_spec import (
    FAMILY_REGISTRY,
    FamilySpec,
)
from quivers.core._util import EPS
from quivers.continuous.bijectors import BIJECTORS_BY_NAME, Bijector
from quivers.continuous.program_steps import Indexed, StepArgument
from quivers.qiec.families import FAMILIES
from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgDist,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
)
from quivers.dsl.family_schemas import (
    FAMILY_ALIASES,
    PARAMETER_ALIASES,
    family_parameterizations,
)
from quivers.dsl.draw_args import bind_family_arguments


class UnsizedParameterError(ValueError):
    """An inline family parameter whose shape only the data would fix.

    The compiled runtime sizes every parameter when it builds a program,
    so a matrix parameter whose extents the checked program leaves open
    cannot be built; the compiler reports it at the draw.
    """


def _floating_support(support: _constraints.Constraint) -> bool:
    """Whether a discrete support takes floating-point values.

    `torch.distributions` scores and samples Boolean (`Bernoulli`) and
    one-hot supports as floating-point tensors; every other discrete
    support (the integer intervals of `Poisson`, `Categorical`,
    `Binomial`, and the rest) as integer tensors.

    Parameters
    ----------
    support : torch.distributions.constraints.Constraint
        The distribution's support.

    Returns
    -------
    bool
        ``True`` for a Boolean or one-hot support.
    """
    return isinstance(support, (_constraints._Boolean, _constraints._OneHot))


def _discrete_sample(
    dist: D.Distribution,
    sample_shape: torch.Size,
    support: _constraints.Constraint,
) -> torch.Tensor:
    """Draw from a discrete distribution at its support's dtype.

    Parameters
    ----------
    dist : torch.distributions.Distribution
        The distribution sampled.
    sample_shape : torch.Size
        Leading sample dimensions.
    support : torch.distributions.constraints.Constraint
        The distribution's support.

    Returns
    -------
    torch.Tensor
        A floating draw for a Boolean or one-hot support, and an
        integer (``long``) draw otherwise.
    """
    draw = dist.sample(sample_shape)
    return draw if _floating_support(support) else draw.long()


def _discrete_value(
    value: torch.Tensor,
    support: _constraints.Constraint,
    floating_dtype: torch.dtype,
) -> torch.Tensor:
    """Cast a value scored by a discrete distribution to its support's dtype.

    A Boolean or one-hot support scores floating values, so an integer
    value (a draw, or integer-typed data) is cast to ``floating_dtype``;
    an integer support scores integer values, so a floating value is
    cast to ``long``. A value already at the right dtype is returned
    unchanged.

    Parameters
    ----------
    value : torch.Tensor
        The value scored.
    support : torch.distributions.constraints.Constraint
        The distribution's support.
    floating_dtype : torch.dtype
        The floating dtype of the distribution's parameters.

    Returns
    -------
    torch.Tensor
        ``value`` at the dtype the support is scored in.
    """
    if _floating_support(support):
        return value if value.dtype.is_floating_point else value.to(floating_dtype)
    return value.long() if value.dtype.is_floating_point else value


def _to_event_shape(value: torch.Tensor, dist: D.Distribution) -> torch.Tensor:
    """Unflatten a matrix-valued value to its distribution's event shape.

    A matrix space such as `CholeskyFactor` lays a value out flat, as
    ``(..., K * K)``, while the torch distribution scores ``(..., K, K)``.

    Parameters
    ----------
    value : torch.Tensor
        The value scored.
    dist : torch.distributions.Distribution
        The distribution scoring it.

    Returns
    -------
    torch.Tensor
        ``value`` with its last axis split into the event shape when it
        holds exactly the event's elements in flat form, and unchanged
        otherwise.
    """
    event = tuple(dist.event_shape)
    if len(event) < 2 or tuple(value.shape[-len(event) :]) == event:
        return value
    if value.dim() >= 1 and value.shape[-1] == math.prod(event):
        return value.reshape(*value.shape[:-1], *event)
    return value


class FixedDistribution(ContinuousMorphism):
    """A distribution with all parameters fixed at construction time.

    The ``rsample`` input is used only for batch size and device
    inference; the distribution parameters themselves are constants.

    Parameters
    ----------
    codomain : AnySpace
        The output space.
    make_dist : callable
        ``(batch_size: int, device: torch.device) -> Distribution``.
    discrete : bool
        Whether the output is discrete (returns LongTensor).
    support : _constraints.Constraint, optional
        The support of the underlying distribution. Used by variational
        guides to apply the right bijector for unconstrained→constrained
        sampling. Defaults to ``constraints.real``; the factory
        functions for each family supply the correct constraint.
    """

    def __init__(
        self,
        codomain: AnySpace,
        make_dist: Callable,
        discrete: bool = False,
        support: _constraints.Constraint | None = None,
        *,
        domain: AnySpace = Unit,
    ) -> None:
        super().__init__(domain, codomain)
        self._make_dist_fn = make_dist
        self._discrete = discrete
        self._support = support if support is not None else D.constraints.real

    def with_domain(self, domain: AnySpace) -> FixedDistribution:
        """Return the same fixed family on a declared input space.

        Parameters
        ----------
        domain : AnySpace
            Input space whose values select no family parameters.

        Returns
        -------
        FixedDistribution
            A fixed distribution that keeps the declared kernel domain.
        """
        return FixedDistribution(
            self.codomain,
            self._make_dist_fn,
            discrete=self._discrete,
            support=self._support,
            domain=domain,
        )

    @property
    def support(self) -> _constraints.Constraint:
        return self._support

    def _get_dist(self, x: torch.Tensor) -> D.Distribution:
        """Build the fixed distribution for an input's batch.

        Parameters
        ----------
        x : torch.Tensor
            Input, used only for its batch size and device.

        Returns
        -------
        torch.distributions.Distribution
            The distribution, with a batch shape of the input's rows.
        """
        return self._make_dist_fn(x.shape[0], x.device)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the fixed distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input (used only for batch size and device).
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples from the distribution.
        """
        batch = x.shape[0]
        dist = self._make_dist_fn(batch, x.device)
        shape = torch.Size(sample_shape)
        if not dist.batch_shape and not dist.event_shape:
            # A closed-form operator measure (the `Restrict` a
            # desugared half-family produces, say) carries no batch of
            # its own: one draw per row, and per coordinate of a
            # continuous codomain, keeps the ``(batch, dim)`` contract
            # of every other fixed distribution.
            rows = (
                torch.Size((batch,))
                if self._discrete
                else torch.Size((batch, int(getattr(self.codomain, "dim", 1))))
            )
            shape = torch.Size((*shape, *rows))
        if self._discrete:
            return _discrete_sample(dist, shape, self._support)
        if getattr(dist, "has_rsample", True):
            return dist.rsample(shape)
        # Continuous but non-reparameterizable (e.g. VonMises).
        return dist.sample(shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability under the fixed distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input (used only for batch size and device).
        y : torch.Tensor
            Output values.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(batch,)``.
        """
        if self._discrete:
            floating = (
                x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
            )
            value = _discrete_value(y, self._support, floating)
        else:
            value = y
        # The distribution's parameters are shared constants, so the
        # density is built once (batch of one) and its constant
        # parameters broadcast against whatever value is scored: a
        # globally-shared prior clamped to a single draw scores once, a
        # per-row batch broadcasts to that batch, and a value carrying
        # the family's own event axis (a Dirichlet simplex, say) keeps
        # that axis as the event rather than mistaking it for a batch.
        # Building at the value's leading axis instead would replicate a
        # shared prior across the program's response count, or read an
        # event coordinate as a batch row.
        device = value.device if value.dim() >= 1 else x.device
        dist = self._make_dist_fn(1, device)
        # A `Measure` (an operator-algebra sub-distribution such as
        # the `Restrict` a desugared half-family produces) carries its
        # residual mass symbolically in ``log_normalizer``; the
        # sample / observe boundary is where the algebra collapses it
        # to a probability density, so we score the renormalised
        # log-density here rather than the raw sub-measure value.
        value = _to_event_shape(value, dist)
        if isinstance(dist, _Measure):
            lp = dist.log_prob_normalized(value)
        else:
            lp = dist.log_prob(value)
        if lp.dim() > 1:
            return lp.sum(dim=-1)
        return lp


class MixedInlineDistribution(ContinuousMorphism):
    """General inline distribution with arbitrary literal/variable mix.

    Handles any combination of literal float values and runtime variable
    inputs for any distribution family. At construction time, receives a
    specification of which parameter positions are fixed literals vs.
    variable inputs, plus a builder function that creates the PyTorch
    distribution from fully-resolved parameter tensors.

    This is the general mechanism underlying all inline distribution
    patterns in the DSL::

        draw x ~ Normal(0.0, 1.0)           # all fixed (0 variable inputs)
        draw y ~ Normal(mu, sigma)           # all variable (2 variable inputs)
        draw z ~ Normal(mu, 0.5)             # mixed (1 variable input)
        draw w ~ TruncatedNormal(mu, 0.5, 0.0, 1.0)  # mixed

    Parameters
    ----------
    domain : AnySpace
        Source space (stacked variable parameters from program env).
    codomain : AnySpace
        Target space.
    param_spec : list of tuple
        For each distribution parameter position, one of:

        - ``('var', dim)``: a variable read from the input, ``dim``
          columns wide, or ``None`` when its width is to be read off
          the stacked input (see ``_variable_widths``);
        - ``('lit', value)``: a fixed literal float value;
        - ``('const', tensor)``: a fixed literal tensor, a list literal
          such as a cutpoint vector, which every row shares.
    dist_builder : callable
        ``(list[torch.Tensor]) -> torch.distributions.Distribution``.
        Receives one 1-D tensor per parameter (all same batch size).
    discrete : bool
        Whether the output is discrete (returns LongTensor).
    support : Constraint, optional
        Support of the underlying distribution; used by variational
        guides. Defaults to ``constraints.real``.
    param_event_ranks : tuple[int, ...], optional
        The rank of one value of each parameter: 0 for a per-row
        scalar, 1 for a vector, 2 for a matrix. Defaults to every
        parameter a scalar.
    param_shapes : tuple[tuple[int, ...] | None, ...], optional
        For each parameter of rank two or more read from the input, the
        shape one row's value has, which its flat columns are laid out
        in; ``None`` elsewhere. Defaults to none.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        param_spec: list[tuple[str, int | float | torch.Tensor | None]],
        dist_builder: Callable,
        discrete: bool = False,
        support: _constraints.Constraint | None = None,
        param_event_ranks: tuple[int, ...] | None = None,
        param_shapes: tuple[tuple[int, ...] | None, ...] | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._param_spec = param_spec
        self._dist_builder = dist_builder
        self._discrete = discrete
        self._support = support if support is not None else _constraints.real
        # Per-position event rank. Position 0 = per-row scalar; >= 1 =
        # vector-shaped distribution parameter (cutpoints, mixture
        # weights / locations / scales). Each consumes the number of
        # columns its ``param_spec`` width records, so the stacked input
        # is split by offset.
        self._param_event_ranks: tuple[int, ...] = (
            param_event_ranks
            if param_event_ranks is not None
            else tuple(0 for _ in param_spec)
        )
        if len(self._param_event_ranks) != len(param_spec):
            raise ValueError(
                "MixedInlineDistribution: param_event_ranks length "
                f"{len(self._param_event_ranks)} disagrees with "
                f"param_spec length {len(param_spec)}"
            )
        self._param_shapes: tuple[tuple[int, ...] | None, ...] = (
            param_shapes
            if param_shapes is not None
            else tuple(None for _ in param_spec)
        )
        if len(self._param_shapes) != len(param_spec):
            raise ValueError(
                "MixedInlineDistribution: param_shapes length "
                f"{len(self._param_shapes)} disagrees with "
                f"param_spec length {len(param_spec)}"
            )
        for (kind, width), rank, shape in zip(
            param_spec, self._param_event_ranks, self._param_shapes, strict=True
        ):
            if kind != "var" or rank < 2:
                continue
            if shape is None or len(shape) != rank or math.prod(shape) != width:
                raise ValueError(
                    "MixedInlineDistribution: a parameter of rank "
                    f"{rank} read from the input needs its shape, of "
                    f"{width} entries; got {shape!r}"
                )

    @property
    def support(self) -> _constraints.Constraint:
        return self._support

    def with_domain(self, domain: AnySpace) -> MixedInlineDistribution:
        """Return the same all-literal family on a declared input space.

        Parameters
        ----------
        domain : AnySpace
            Input space whose values select no family parameters.

        Returns
        -------
        MixedInlineDistribution
            The family with every parameter a literal, reading its
            input only for the number of rows.

        Raises
        ------
        ValueError
            If a parameter is read from the input, whose columns the
            new domain would not carry.
        """
        if any(kind == "var" for kind, _ in self._param_spec):
            raise ValueError(
                "MixedInlineDistribution.with_domain: a parameter is read from "
                "the input, so the input space is fixed by the parameters"
            )
        return MixedInlineDistribution(
            domain,
            self.codomain,
            self._param_spec,
            self._dist_builder,
            discrete=self._discrete,
            support=self._support,
            param_event_ranks=self._param_event_ranks,
        )

    def _resolve_params(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Reconstruct full parameter list from input + stored literals.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters. Shape ``(batch, total_var_dim)``.
            May be ``(batch,)`` if total variable dimension is 1. When
            the family declares a vector-typed final parameter, the
            trailing columns of ``x`` are the vector values.

        Returns
        -------
        list of torch.Tensor
            One tensor per parameter, each shape ``(batch,)`` or
            ``(batch, dim)``.
        """
        if x.dim() == 0:
            # One run's scalar parameter: a single row of one column.
            x = x.reshape(1, 1)
        if x.dim() == 1:
            x = x.unsqueeze(0) if self._is_shared_vector(x) else x.unsqueeze(-1)
        positions = [
            pos for pos, (kind, _) in enumerate(self._param_spec) if kind == "var"
        ]
        # With every parameter a literal the input carries no parameter
        # columns; it only counts the rows.
        widths = self._variable_widths(x.shape[-1]) if positions else []
        variables: list[torch.Tensor] = []
        batch_ranks: list[int] = []
        var_offset = 0
        for pos, dim in zip(positions, widths, strict=True):
            event_rank = self._param_event_ranks[pos]
            shape = self._param_shapes[pos]
            if dim == 1 and event_rank < 1:
                variables.append(x[..., var_offset])
            elif event_rank >= 2:
                # A matrix parameter's columns are its entries in row-major
                # order, one row's value per row of the input.
                assert shape is not None
                block = x[..., var_offset : var_offset + dim]
                variables.append(block.reshape(*block.shape[:-1], *shape))
            else:
                variables.append(x[..., var_offset : var_offset + dim])
            batch_ranks.append(
                variables[-1].dim() - (event_rank if event_rank >= 2 else 0)
            )
            var_offset += dim
        # A literal broadcasts against the widest variable parameter: a
        # per-row value, with a unit axis for each further axis a scalar
        # or vector parameter carries. A matrix parameter's own axes are
        # its value's, which a literal does not broadcast along.
        rank = max(batch_ranks, default=1)
        literal_shape = (x.shape[0],) + (1,) * (rank - 1)
        params: list[torch.Tensor] = []
        supplied = iter(variables)
        for kind, value in self._param_spec:
            if kind == "lit":
                assert isinstance(value, int | float)
                params.append(
                    torch.full(
                        literal_shape, float(value), device=x.device, dtype=x.dtype
                    )
                )
            elif kind == "const":
                assert isinstance(value, torch.Tensor)
                constant = value.to(device=x.device, dtype=x.dtype)
                params.append(constant.expand(x.shape[0], *constant.shape))
            else:
                params.append(next(supplied))
        return params

    def _is_shared_vector(self, x: torch.Tensor) -> bool:
        """Whether a 1-D input is one shared vector rather than a column.

        A family whose one variable parameter is a vector reads a 1-D
        input of that vector's declared width as a single row, as the
        stacking of program arguments does for a shared ``(dim,)``
        vector; any other 1-D input is one scalar per row.

        Parameters
        ----------
        x : torch.Tensor
            A 1-D input.

        Returns
        -------
        bool
            Whether ``x`` is the one vector parameter's value.
        """
        variables = [
            (value, rank)
            for (kind, value), rank in zip(
                self._param_spec, self._param_event_ranks, strict=True
            )
            if kind == "var"
        ]
        if len(variables) != 1:
            return False
        width, rank = variables[0]
        return rank >= 1 and width is not None and int(width) == x.shape[0] > 1

    def _variable_widths(self, stacked: int) -> list[int]:
        """The column count of each variable parameter.

        A declared width is taken as given. A parameter whose binding
        had no known type leaves its width unresolved (``None``), and
        the stacked width resolves it: beside one unresolved vector
        parameter, each unresolved scalar parameter is one column per
        row and the vector takes the rest; otherwise one unresolved
        scalar takes every remaining column, and several take an equal
        share when that share is one column per row or one per
        coordinate of the codomain. The columns must be
        accounted for exactly, so a mis-declared width is an error
        rather than a silent shift of columns between parameters.

        Parameters
        ----------
        stacked : int
            The width of the stacked variable parameters.

        Returns
        -------
        list[int]
            One width per ``var`` entry of the parameter specification.

        Raises
        ------
        ValueError
            If the declared widths do not account for the stacked width,
            or the unresolved widths admit no single reading.
        """
        variables = [
            (value, rank)
            for (kind, value), rank in zip(
                self._param_spec, self._param_event_ranks, strict=True
            )
            if kind == "var"
        ]
        declared = [width for width, _ in variables]
        known = sum(int(width) for width in declared if width is not None)
        remainder = stacked - known
        open_scalars = sum(1 for w, r in variables if w is None and r < 1)
        open_vectors = sum(1 for w, r in variables if w is None and r >= 1)
        if not open_scalars and not open_vectors:
            if remainder != 0:
                raise ValueError(
                    "MixedInlineDistribution: the stacked parameters have "
                    f"{stacked} columns, but the declared widths sum to {known}; "
                    "a parameter's declared width disagrees with its value"
                )
            return [int(width) for width in declared if width is not None]
        coordinates = int(getattr(self.codomain, "dim", 1))
        scalar_share: int
        vector_share = 0
        if open_vectors == 1:
            # Beside a vector parameter, a scalar one is one value per
            # row; the vector takes the rest.
            scalar_share = 1
            vector_share = remainder - open_scalars
        elif open_vectors == 0 and open_scalars == 1 and remainder >= 1:
            scalar_share = remainder
        elif open_vectors == 0 and remainder in (
            open_scalars,
            open_scalars * coordinates,
        ):
            scalar_share = remainder // open_scalars
        else:
            scalar_share = 0
        if (open_scalars and scalar_share < 1) or (open_vectors and vector_share < 1):
            raise ValueError(
                "MixedInlineDistribution: the stacked parameters have "
                f"{stacked} columns; after the declared widths ({known}) the "
                f"{remainder} remaining cannot be divided among "
                f"{open_scalars + open_vectors} parameters of unknown width as "
                "one column per row or one per coordinate of the "
                f"{coordinates}-dimensional codomain"
            )
        return [
            int(width)
            if width is not None
            else (vector_share if rank >= 1 else scalar_share)
            for width, rank in variables
        ]

    def _get_dist(self, x: torch.Tensor) -> D.Distribution:
        """Build the distribution at an input.

        Parameters
        ----------
        x : torch.Tensor
            The stacked inline arguments.

        Returns
        -------
        torch.distributions.Distribution
            The distribution the builder makes of the resolved
            parameters.
        """
        return self._dist_builder(self._resolve_params(x))

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from the distribution.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters.
        sample_shape : torch.Size
            Additional leading sample dimensions.

        Returns
        -------
        torch.Tensor
            Samples from the distribution.
        """
        params = self._resolve_params(x)
        dist = self._dist_builder(params)
        if self._discrete:
            # Discrete samples are kept at the distribution's native
            # batch shape; downstream code unpacks as needed.
            return _discrete_sample(dist, torch.Size(sample_shape), self._support)
        if getattr(dist, "has_rsample", True):
            result = dist.rsample(sample_shape)
        else:
            # Continuous but non-reparameterizable (e.g. VonMises).
            result = dist.sample(sample_shape)
        if result.dim() == 1:
            result = result.unsqueeze(-1)
        return result

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability under the distribution.

        Parameters
        ----------
        x : torch.Tensor
            Stacked variable parameters.
        y : torch.Tensor
            Observed values.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape matches the broadcast of the
            distribution's batch shape with ``y``'s shape, with any
            event dimension summed out.
        """
        params = self._resolve_params(x)
        dist = self._dist_builder(params)
        # A discrete value is scored at its support's dtype: floating
        # for a Boolean or one-hot support, integer otherwise, whether
        # it is observed data or a draw of the site.
        if self._discrete:
            floating = next(
                (p.dtype for p in params if p.dtype.is_floating_point),
                torch.get_default_dtype(),
            )
            y_in = _discrete_value(y, self._support, floating)
        else:
            y_in = y
        # Reconcile y's shape with the distribution's expected
        # event shape: append a trailing singleton if the
        # distribution carries an event dim that y lacks; drop a
        # trailing singleton from y when the distribution is scalar
        # and y carries an extra unit dim from upstream reshapes.
        y_in = _to_event_shape(y_in, dist)
        event_dim = len(dist.event_shape)
        batch_dim = len(dist.batch_shape)
        target_dim = event_dim + batch_dim
        if not self._discrete:
            while y_in.dim() < target_dim:
                # A value holding one whole event lacks the batch axes;
                # any other lacks a trailing coordinate axis.
                whole_event = event_dim > 0 and tuple(
                    y_in.shape[y_in.dim() - event_dim :]
                ) == tuple(dist.event_shape)
                y_in = y_in.unsqueeze(0) if whole_event else y_in.unsqueeze(-1)
            while y_in.dim() > target_dim and y_in.shape[-1] == 1:
                y_in = y_in.squeeze(-1)
        lp = dist.log_prob(y_in)
        # Scalar parameters read one value per coordinate make the
        # coordinates independent draws of one value: their densities
        # sum, as a fixed distribution's do, leaving one per row.
        coordinates = self._scalar_coordinates(params)
        if coordinates > 1 and lp.dim() >= 1 and lp.shape[-1] == coordinates:
            return lp.sum(dim=-1)
        return lp

    def _scalar_coordinates(self, params: list[torch.Tensor]) -> int:
        """How many coordinates per row the scalar parameters carry.

        Parameters
        ----------
        params : list[torch.Tensor]
            The resolved parameters.

        Returns
        -------
        int
            The widest trailing axis among the variable scalar
            parameters read as per-coordinate blocks, or one.
        """
        widths = [
            int(t.shape[-1])
            for t, (kind, _), rank in zip(
                params, self._param_spec, self._param_event_ranks, strict=True
            )
            if kind == "var" and rank < 1 and t.dim() >= 2
        ]
        return max(widths, default=1)


def make_fixed_logitnormal(
    mu: float, sigma: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LogitNormal(mu, sigma) distribution.

    Parameters
    ----------
    mu : float
        Location parameter of the underlying normal.
    sigma : float
        Scale parameter of the underlying normal (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from LogitNormal(mu, sigma).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        mu_t = torch.full((batch, d), mu, device=device)
        sigma_t = torch.full((batch, d), sigma, device=device)
        base = D.Normal(mu_t, sigma_t)
        return D.TransformedDistribution(base, [D.SigmoidTransform()])

    return FixedDistribution(codomain, builder, support=_constraints.unit_interval)


def make_fixed_uniform(
    low: float, high: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Uniform(low, high) distribution.

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Uniform(low, high).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        low_t = torch.full((batch, d), low, device=device)
        high_t = torch.full((batch, d), high, device=device)
        return D.Uniform(low_t, high_t)

    return FixedDistribution(
        codomain, builder, support=_constraints.interval(low, high)
    )


def make_fixed_normal(
    loc: float, scale: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Normal(loc, scale) distribution.

    Parameters
    ----------
    loc : float
        Mean.
    scale : float
        Standard deviation (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Normal(loc, scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        loc_t = torch.full((batch, d), loc, device=device)
        scale_t = torch.full((batch, d), scale, device=device)
        return D.Normal(loc_t, scale_t)

    return FixedDistribution(codomain, builder)


def make_fixed_bernoulli(prob: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed Bernoulli(prob) distribution.

    Parameters
    ----------
    prob : float
        Success probability.
    codomain : AnySpace
        Output space (FinSet of size 2).

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Bernoulli(prob).
    """

    def builder(batch: int, device: torch.device) -> D.Distribution:
        probs_t = torch.full((batch,), prob, device=device)
        return D.Bernoulli(probs=probs_t)

    return FixedDistribution(
        codomain, builder, discrete=True, support=_constraints.boolean
    )


def make_fixed_beta(
    concentration1: float, concentration0: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Beta(concentration1, concentration0) distribution.

    Parameters
    ----------
    concentration1 : float
        Alpha parameter.
    concentration0 : float
        Beta parameter.
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Beta(a, b).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        a = torch.full((batch, d), concentration1, device=device)
        b = torch.full((batch, d), concentration0, device=device)
        return D.Beta(a, b)

    return FixedDistribution(codomain, builder, support=_constraints.unit_interval)


def make_fixed_logit_beta(
    concentration1: float, concentration0: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LogitBeta(concentration1, concentration0) distribution.

    The draw is ``logit(p)`` for ``p ~ Beta(concentration1,
    concentration0)``, scored and sampled on the real line by
    [`LogitBeta`][quivers.continuous.log_space.LogitBeta]. The shapes
    are passed through unclamped, so a shape far below one keeps its
    exact density rather than being lifted to a floor.

    Parameters
    ----------
    concentration1 : float
        The Beta shape on ``p`` (positive).
    concentration0 : float
        The Beta shape on ``1 - p`` (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from LogitBeta(a, b).

    Raises
    ------
    ValueError
        If either shape is not positive.
    """
    _require_positive("LogitBeta", concentration1=concentration1)
    _require_positive("LogitBeta", concentration0=concentration0)
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        a = torch.full((batch, d), concentration1, device=device)
        b = torch.full((batch, d), concentration0, device=device)
        return LogitBeta(a, b)

    return FixedDistribution(codomain, builder, support=_constraints.real)


def make_fixed_log_gamma(
    concentration: float, rate: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LogGamma(concentration, rate) distribution.

    The draw is ``log(g)`` for ``g ~ Gamma(concentration, rate)``,
    scored and sampled on the real line by
    [`LogGamma`][quivers.continuous.log_space.LogGamma]. The parameters
    are passed through unclamped, so a shape far below one keeps its
    exact density rather than being lifted to a floor.

    Parameters
    ----------
    concentration : float
        The Gamma shape (positive).
    rate : float
        The Gamma rate (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from LogGamma(concentration, rate).

    Raises
    ------
    ValueError
        If the shape or the rate is not positive.
    """
    _require_positive("LogGamma", concentration=concentration)
    _require_positive("LogGamma", rate=rate)
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        conc_t = torch.full((batch, d), concentration, device=device)
        rate_t = torch.full((batch, d), rate, device=device)
        return LogGamma(conc_t, rate_t)

    return FixedDistribution(codomain, builder, support=_constraints.real)


def _require_positive(family: str, **parameter: float) -> None:
    """Refuse a literal parameter that is not a positive number.

    Parameters
    ----------
    family : str
        The family, for the diagnostic.
    **parameter : float
        One keyword naming the parameter and giving its value.

    Raises
    ------
    ValueError
        If the value is not positive.
    """
    for name, value in parameter.items():
        if not float(value) > 0.0:
            raise ValueError(f"{family} needs a positive {name}; got {value!r}")


def make_fixed_exponential(rate: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed Exponential(rate) distribution.

    Parameters
    ----------
    rate : float
        Rate parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Exponential(rate).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        rate_t = torch.full((batch, d), rate, device=device)
        return D.Exponential(rate_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_halfcauchy(scale: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed HalfCauchy(scale) distribution.

    Parameters
    ----------
    scale : float
        Scale parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from HalfCauchy(scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        scale_t = torch.full((batch, d), scale, device=device)
        return D.HalfCauchy(scale_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_halfnormal(scale: float, codomain: AnySpace) -> FixedDistribution:
    """Create a fixed HalfNormal(scale) distribution.

    Parameters
    ----------
    scale : float
        Scale parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from HalfNormal(scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        scale_t = torch.full((batch, d), scale, device=device)
        return D.HalfNormal(scale_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_lognormal(
    loc: float, scale: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LogNormal(loc, scale) distribution.

    Parameters
    ----------
    loc : float
        Mean of the underlying normal.
    scale : float
        Standard deviation of the underlying normal (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from LogNormal(loc, scale).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        loc_t = torch.full((batch, d), loc, device=device)
        scale_t = torch.full((batch, d), scale, device=device)
        return D.LogNormal(loc_t, scale_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_gamma(
    concentration: float, rate: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Gamma(concentration, rate) distribution.

    Parameters
    ----------
    concentration : float
        Shape parameter (positive).
    rate : float
        Rate parameter (positive).
    codomain : AnySpace
        Output space.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from Gamma(concentration, rate).
    """
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        conc_t = torch.full((batch, d), concentration, device=device)
        rate_t = torch.full((batch, d), rate, device=device)
        return D.Gamma(conc_t, rate_t)

    return FixedDistribution(codomain, builder, support=_constraints.positive)


def make_fixed_dirichlet(
    concentration: float | list[float], codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed Dirichlet(concentration) distribution on the simplex
    over the codomain's element axis.

    Parameters
    ----------
    concentration : float or list of float
        Concentration parameter. A scalar is broadcast to all simplex
        components (symmetric Dirichlet). A list / tuple of length
        equal to the codomain's dimension specifies per-component
        concentrations. Each entry must be positive.
    codomain : AnySpace
        Output space; its ``dim`` attribute gives the simplex
        dimension (the number of components).

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling from the chosen Dirichlet on
        the (d-1)-simplex embedded in ``R^d``.
    """
    d = getattr(codomain, "dim", 1)
    if d < 2:
        raise ValueError(f"Dirichlet codomain must have dim >= 2, got dim={d}")
    if isinstance(concentration, (list, tuple)):
        if len(concentration) == 1:
            # Single-element sequence is treated as a scalar
            # (symmetric Dirichlet); broadcast to the codomain's
            # simplex dimension.
            conc_values = [float(concentration[0])] * d
        elif len(concentration) != d:
            raise ValueError(
                f"Dirichlet concentration vector has length "
                f"{len(concentration)} but codomain has dim={d}"
            )
        else:
            conc_values = [float(c) for c in concentration]
    else:
        conc_values = [float(concentration)] * d
    if any(c <= 0.0 for c in conc_values):
        raise ValueError(
            f"Dirichlet concentration must be positive componentwise, got {conc_values}"
        )
    conc_tuple = tuple(conc_values)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        conc_t = torch.tensor(conc_tuple, device=device).expand(batch, d)
        return D.Dirichlet(conc_t)

    return FixedDistribution(codomain, builder, support=_constraints.simplex)


def make_fixed_truncated_normal(
    mu: float,
    sigma: float,
    low: float,
    high: float,
    codomain: AnySpace,
) -> FixedDistribution:
    """Create a fixed truncated normal with literal parameters."""
    d = getattr(codomain, "dim", 1)

    def builder(batch: int, device: torch.device) -> D.Distribution:
        params = [
            torch.full((batch, d), value, device=device)
            for value in (mu, sigma, low, high)
        ]
        return _truncated_normal_builder(params)

    return FixedDistribution(
        codomain,
        builder,
        support=_constraints.interval(low, high),
    )


def make_fixed_categorical(
    values: float | list[float],
    codomain: AnySpace,
    *,
    parameterization: str,
) -> FixedDistribution:
    """Create a literal categorical from probabilities or logits."""
    cardinality = int(getattr(codomain, "cardinality", getattr(codomain, "size", 2)))
    if isinstance(values, (list, tuple)):
        vector = [float(value) for value in values]
        if len(vector) != cardinality:
            raise ValueError(
                f"Categorical {parameterization} vector has length "
                f"{len(vector)} but codomain has cardinality={cardinality}"
            )
    else:
        vector = [float(values)] * cardinality

    def builder(batch: int, device: torch.device) -> D.Distribution:
        parameter = torch.tensor(vector, device=device).expand(batch, cardinality)
        if parameterization == "logits":
            return D.Categorical(logits=parameter)
        return D.Categorical(probs=parameter)

    return FixedDistribution(
        codomain,
        builder,
        discrete=True,
        support=_constraints.integer_interval(0, cardinality - 1),
    )


def make_fixed_lkj_cholesky(
    concentration: float, codomain: AnySpace
) -> FixedDistribution:
    """Create a fixed LKJ prior over Cholesky factors of correlation matrices.

    Parameters
    ----------
    concentration : float
        The LKJ shape, positive; one is uniform over correlation
        matrices and larger values concentrate toward the identity.
    codomain : AnySpace
        Output space; its ``cardinality`` or ``dim`` gives the
        correlation matrix's size, as for the variable-parameter form.

    Returns
    -------
    FixedDistribution
        Distribution morphism sampling lower-triangular Cholesky factors.

    Raises
    ------
    ValueError
        If ``concentration`` is not positive, or the codomain does not
        fix a matrix size of at least two.
    """
    if concentration <= 0.0:
        raise ValueError(
            f"LKJCholesky concentration must be positive, got {concentration}"
        )
    dim = _codomain_dim_or_raise("LKJCholesky", codomain)
    if dim < 2:
        raise ValueError(f"LKJCholesky needs a matrix size of at least 2, got {dim}")

    def builder(batch: int, device: torch.device) -> D.Distribution:
        shape = torch.full((batch,), float(concentration), device=device)
        return OnionLKJCholesky(dim, shape)

    return FixedDistribution(codomain, builder, support=_constraints.corr_cholesky)


# Families whose all-literal factory takes a single vector
# argument (a ``list[float]`` or ``tuple[float, ...]``) rather than
# one positional float per scalar parameter. The inline call site in
# ``make_inline_distribution`` re-bundles the splat-flattened
# literals into a list before invoking the factory.
_VECTOR_PARAM_FAMILIES: frozenset[str] = frozenset({"Dirichlet"})


_FIXED_FACTORIES: dict[str, tuple[tuple[str, ...], Callable]] = {
    "LogitNormal": (("mu", "sigma"), make_fixed_logitnormal),
    "Normal": (("loc", "scale"), make_fixed_normal),
    "Uniform": (("low", "high"), make_fixed_uniform),
    "Bernoulli": (("probs",), make_fixed_bernoulli),
    "Beta": (("concentration1", "concentration0"), make_fixed_beta),
    "Exponential": (("rate",), make_fixed_exponential),
    "HalfCauchy": (("scale",), make_fixed_halfcauchy),
    "HalfNormal": (("scale",), make_fixed_halfnormal),
    "LogNormal": (("loc", "scale"), make_fixed_lognormal),
    "Gamma": (("concentration", "rate"), make_fixed_gamma),
    "LogGamma": (("concentration", "rate"), make_fixed_log_gamma),
    "LogitBeta": (("concentration1", "concentration0"), make_fixed_logit_beta),
    "Dirichlet": (("concentration",), make_fixed_dirichlet),
    "LKJCholesky": (("concentration",), make_fixed_lkj_cholesky),
    "TruncatedNormal": (
        ("mu", "sigma", "low", "high"),
        make_fixed_truncated_normal,
    ),
}

# Per-family support constraints for inline distributions. Used when
# constructing a `MixedInlineDistribution` (which has at least one
# variable-bound parameter) so the resulting morphism advertises the
# correct constrained support to variational guides.
_FAMILY_SUPPORTS: dict[str, _constraints.Constraint] = {
    "Normal": _constraints.real,
    "Bernoulli": _constraints.boolean,
    "TruncatedNormal": _constraints.real,  # interval is set per-call
    "LogitNormal": _constraints.unit_interval,
    "Uniform": _constraints.real,  # interval is set per-call
    "Beta": _constraints.unit_interval,
    "Exponential": _constraints.positive,
    "HalfCauchy": _constraints.positive,
    "HalfNormal": _constraints.positive,
    "LogNormal": _constraints.positive,
    "Gamma": _constraints.positive,
    "LogGamma": _constraints.real,
    "LogitBeta": _constraints.real,
    "Dirichlet": _constraints.simplex,
    "Logistic": _constraints.real,
    "HalfStudentT": _constraints.positive,
    "BetaBinomial": _constraints.nonnegative_integer,
    "OrderedLogistic": _constraints.nonnegative_integer,
    "OrderedProbit": _constraints.nonnegative_integer,
    "InverseWishart": _constraints.positive_definite,
    "MatrixNormal": _constraints.independent(_constraints.real, 2),
    "ZeroInflatedPoisson": _constraints.nonnegative_integer,
    "HurdlePoisson": _constraints.nonnegative_integer,
    "ZeroOneInflatedBeta": _constraints.unit_interval,
    "MixtureNormal": _constraints.real,
}


_OPERATOR_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "PointMass": ("value",),
    "Restrict": ("base", "low", "high"),
    "Truncate": ("base", "low", "high"),
    "Pushforward": ("base", "bijector"),
    "Mixture": ("weights", "components"),
    "Independent": ("base", "reinterpreted_batch_ndims"),
    "Normalize": ("base",),
}


def get_inline_param_names(family: str) -> tuple[str, ...] | None:
    """Get the ordered parameter names for an inline family.

    Parameters
    ----------
    family : str
        Distribution family name.

    Returns
    -------
    tuple[str, ...] or None
        Parameter names, or None if not an inline family.
    """
    if family in _OPERATOR_PARAM_NAMES:
        return _OPERATOR_PARAM_NAMES[family]
    if family in _FIXED_FACTORIES:
        return _FIXED_FACTORIES[family][0]
    if family in _FAMILY_BUILDERS:
        return _FAMILY_BUILDERS[family][0]
    if family == "TruncatedNormal":
        return ("mu", "sigma", "low", "high")
    return None


def get_inline_parameterizations(family: str) -> tuple[tuple[str, ...], ...]:
    """Return every complete keyword schema accepted by a family call.

    Most families expose one canonical schema. ``Categorical`` exposes two
    one-parameter schemas so probabilities and logits occupy the same
    source-language level without overloading a positional value.
    """
    canonical = get_inline_param_names(family)
    if canonical is None:
        return ()
    return family_parameterizations(family, canonical)


def _normal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Normal from [loc, scale]."""
    return D.Normal(params[0], params[1].clamp(min=EPS))


def _bernoulli_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Bernoulli from [probs]."""
    return D.Bernoulli(probs=params[0].clamp(EPS, 1.0 - EPS))


def _truncated_normal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build TruncatedNormal from [mu, sigma, low, high].

    Returns a proxy object with rsample and log_prob that performs
    inverse-CDF truncated normal sampling.
    """
    mu, sigma, low_t, high_t = params
    sigma = sigma.clamp(min=EPS)
    low = float(low_t.flatten()[0])
    high = float(high_t.flatten()[0])

    class _TruncNorm:
        """Minimal truncated-normal distribution interface.

        Exposes the same ``event_shape`` and ``batch_shape``
        attributes the generic `torch.distributions` interface
        carries, so callers walking the family registry's
        ``log_prob`` path treat it uniformly with stock families.
        """

        event_shape: torch.Size = torch.Size()
        batch_shape: torch.Size = mu.shape

        def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            normal = D.Normal(0, 1)
            alpha = normal.cdf((low - mu) / sigma)
            beta_cdf = normal.cdf((high - mu) / sigma)
            u = torch.rand(*sample_shape, *mu.shape, device=mu.device, dtype=mu.dtype)
            u_scaled = alpha + u * (beta_cdf - alpha)
            u_scaled = u_scaled.clamp(min=EPS, max=1.0 - EPS)
            return normal.icdf(u_scaled) * sigma + mu

        def log_prob(self, y: torch.Tensor) -> torch.Tensor:
            log_phi = (
                -0.5 * ((y - mu) / sigma) ** 2
                - sigma.log()
                - 0.5 * math.log(2 * math.pi)
            )
            normal = D.Normal(0, 1)
            log_Z = torch.log(
                (
                    normal.cdf((high - mu) / sigma) - normal.cdf((low - mu) / sigma)
                ).clamp(min=EPS)
            )
            return log_phi - log_Z

        def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
            with torch.no_grad():
                return self.rsample(sample_shape)

    return _TruncNorm()


def _logitnormal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build LogitNormal from [mu, sigma]."""
    base = D.Normal(params[0], params[1].clamp(min=EPS))
    return D.TransformedDistribution(base, [D.SigmoidTransform()])


def _logit_beta_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build LogitBeta from [concentration1, concentration0], unclamped.

    The distribution checks that both shapes are positive, so a shape
    that has reached zero raises rather than being lifted to a floor.
    """
    return LogitBeta(params[0], params[1], validate_args=True)


def _log_gamma_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build LogGamma from [concentration, rate], unclamped.

    The distribution checks that both parameters are positive, so a
    parameter that has reached zero raises rather than being lifted to
    a floor.
    """
    return LogGamma(params[0], params[1], validate_args=True)


def _uniform_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Uniform from [low, high]."""
    return D.Uniform(params[0], params[1])


def _beta_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Beta from [concentration1, concentration0]."""
    return D.Beta(params[0].clamp(min=EPS), params[1].clamp(min=EPS))


def _exponential_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Exponential from [rate]."""
    return D.Exponential(params[0].clamp(min=EPS))


def _halfcauchy_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build HalfCauchy from [scale]."""
    return D.HalfCauchy(params[0].clamp(min=EPS))


def _halfnormal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build HalfNormal from [scale]."""
    return D.HalfNormal(params[0].clamp(min=EPS))


def _lognormal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build LogNormal from [loc, scale]."""
    return D.LogNormal(params[0], params[1].clamp(min=EPS))


def _gamma_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Gamma from [concentration, rate]."""
    return D.Gamma(params[0].clamp(min=EPS), params[1].clamp(min=EPS))


def _dirichlet_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build Dirichlet from [concentration]. The concentration tensor is
    ``(batch,)`` for scalar input (broadcast to the simplex dimension at
    sample time) or ``(batch, d)`` for a per-fibre vector."""
    return D.Dirichlet(params[0].clamp(min=EPS))


def _logistic_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build a Logistic distribution from [loc, scale].

    torch does not ship a `Logistic` directly; it is constructed as a
    location-scale transform of the standard logistic, which torch
    exposes as a `TransformedDistribution` of a base Uniform through
    the logit map. The implementation here uses a
    `TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv,
    AffineTransform(loc, scale)])` which has the right density.
    """
    loc = params[0]
    scale = params[1].clamp(min=EPS)
    base = D.Uniform(torch.zeros_like(loc), torch.ones_like(loc))
    return D.TransformedDistribution(
        base,
        [
            D.transforms.SigmoidTransform().inv,
            D.transforms.AffineTransform(loc=loc, scale=scale),
        ],
    )


class _HalfStudentT(D.Distribution):
    """Half-Student-t distribution: the folded form of a centered
    Student-t.

    `log_prob(x) = log 2 + StudentT(df, 0, scale).log_prob(x)` for
    `x >= 0`, and `-inf` otherwise. Mirrors the structure
    `torch.distributions.HalfNormal` uses (override `log_prob`
    directly rather than route through `TransformedDistribution`
    with `AbsTransform`, which lacks `log_abs_det_jacobian`).
    """

    arg_constraints = {
        "df": _constraints.positive,
        "scale": _constraints.positive,
    }
    support = _constraints.positive
    has_rsample = False

    def __init__(
        self,
        df: torch.Tensor,
        scale: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.df = df
        self.scale = scale
        self._base = D.StudentT(df, torch.zeros_like(scale), scale)
        super().__init__(
            batch_shape=self._base.batch_shape,
            validate_args=validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        lp = self._base.log_prob(value) + math.log(2.0)
        return torch.where(value >= 0, lp, torch.full_like(lp, float("-inf")))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._base.sample(sample_shape).abs()


def _half_student_t_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build a half-Student-t distribution from [df, scale].

    Uses the explicit `_HalfStudentT` class (which overrides
    `log_prob` with the folded identity) rather than
    `TransformedDistribution(StudentT, AbsTransform)`, because
    torch's `AbsTransform` does not implement
    `log_abs_det_jacobian` and the transformed-distribution
    `log_prob` would crash with `NotImplementedError`.
    """
    df = params[0].clamp(min=EPS)
    scale = params[1].clamp(min=EPS)
    return _HalfStudentT(df, scale)


class _BetaBinomial(D.Distribution):
    """Beta-Binomial: ``Binomial(n, p)`` with ``p ~ Beta(a, b)``.

    The marginal pmf is

        p(k; n, a, b) = C(n, k) * B(a + k, b + n - k) / B(a, b),

    where ``B(.,.)`` is the beta function. `log_prob` evaluates this
    in log space through `torch.lgamma`; `sample` draws ``p ~ Beta(a,
    b)`` then ``k ~ Binomial(n, p)``, matching the generative
    definition.

    Defined here in the continuous layer so the inline builder carries
    no dependency on the transpile layer.
    """

    arg_constraints = {
        "total_count": _constraints.nonnegative_integer,
        "concentration1": _constraints.positive,
        "concentration0": _constraints.positive,
    }
    support = _constraints.nonnegative_integer
    has_rsample = False

    def __init__(
        self,
        total_count: torch.Tensor,
        concentration1: torch.Tensor,
        concentration0: torch.Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.total_count = total_count
        self.concentration1 = concentration1
        self.concentration0 = concentration0
        super().__init__(validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """``log p(value; n, a, b)`` via the closed-form Beta-Binomial pmf.

        The calculation promotes all inputs to a floating result dtype
        and evaluates the ``lgamma`` terms in ``float64`` to limit
        cancellation at large trial counts.
        """
        dtype = torch.promote_types(
            torch.promote_types(
                torch.promote_types(self.total_count.dtype, self.concentration1.dtype),
                self.concentration0.dtype,
            ),
            torch.promote_types(value.dtype, torch.get_default_dtype()),
        )
        n = self.total_count.double()
        a = self.concentration1.double()
        b = self.concentration0.double()
        k = value.double()
        log_comb = (
            torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(n - k + 1.0)
        )
        log_beta_post = (
            torch.lgamma(a + k) + torch.lgamma(b + n - k) - torch.lgamma(a + b + n)
        )
        log_beta_prior = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        return (log_comb + log_beta_post - log_beta_prior).to(dtype)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Two-stage draw: ``p ~ Beta(a, b)``; ``k ~ Binomial(n, p)``."""
        p = D.Beta(self.concentration1, self.concentration0).sample(sample_shape)
        return D.Binomial(total_count=self.total_count, probs=p).sample()

    @property
    def mean(self) -> torch.Tensor:
        """``E[K] = n * a / (a + b)``."""
        return (
            self.total_count
            * self.concentration1
            / (self.concentration1 + self.concentration0)
        )

    @property
    def variance(self) -> torch.Tensor:
        """Closed-form Beta-Binomial variance."""
        n = self.total_count
        a = self.concentration1
        b = self.concentration0
        return n * a * b * (a + b + n) / ((a + b) ** 2 * (a + b + 1.0))


def _beta_binomial_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build a Beta-Binomial distribution from [total_count,
    concentration1, concentration0].

    The distribution is the continuous-layer `_BetaBinomial`, whose
    closed-form `log_prob` exposes `sample`, `mean`, and `variance`
    directly and keeps the inline path free of a transpile import.
    """
    total_count = params[0].clamp(min=EPS)
    a = params[1].clamp(min=EPS)
    b = params[2].clamp(min=EPS)
    return _BetaBinomial(total_count, a, b)


_DIM_DEPENDENT_FAMILIES: frozenset[str] = frozenset({"LKJCholesky"})


def _dim_dependent_builder(
    family: str, codomain: AnySpace
) -> Callable[[list[torch.Tensor]], D.Distribution]:
    """Return a dim-aware builder closure for a family whose torch
    distribution needs the matrix / vector dimension at construction
    time.

    Currently used for `LKJCholesky`, whose ``dim`` argument fixes the
    Cholesky factor's shape; the closure resolves the dim from the
    codomain (FinSet cardinality or Euclidean / CholeskyFactor dim)
    and embeds it.
    """
    dim = _codomain_dim_or_raise(family, codomain)
    if family == "LKJCholesky":

        def _builder(params: list[torch.Tensor]) -> D.Distribution:
            concentration = params[0].clamp(min=EPS)
            return OnionLKJCholesky(dim, concentration)

        return _builder
    raise ValueError(
        f"_dim_dependent_builder: family {family!r} is marked "
        "dim-dependent but no closure factory is registered"
    )


def _codomain_dim_or_raise(family: str, codomain: AnySpace) -> int:
    """Resolve the matrix / vector dimension a dim-dependent family
    requires from its ``codomain``.

    Supported codomain shapes: anything that exposes a ``cardinality``
    field (the FinSet case for ``object Dim : FinSet K``), a ``dim``
    field (Euclidean / CholeskyFactor / Simplex / PositiveReals), or a
    callable ``shape`` property reducing to a single integer.

    Raises `ValueError` rather than silently coercing to a wrong dim,
    because a wrong LKJCholesky dim is the kind of bug that produces a
    numerically valid but semantically wrong density.
    """
    card = getattr(codomain, "cardinality", None)
    if isinstance(card, int):
        return card
    dim_attr = getattr(codomain, "dim", None)
    if isinstance(dim_attr, int):
        return dim_attr
    raise ValueError(
        f"_codomain_dim_or_raise: cannot resolve matrix dimension "
        f"for inline family {family!r} from codomain "
        f"{type(codomain).__name__} (no `cardinality` or `dim` "
        "attribute). Declare the codomain with an explicit size "
        "(e.g., `object Dim : FinSet 4`) or extend "
        "`_codomain_dim_or_raise` to handle this space kind."
    )


def _zero_inflated_poisson_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `ZeroInflatedPoisson` from ``[zero_prob, rate]``."""
    return ZeroInflatedPoisson(
        params[0].clamp(EPS, 1.0 - EPS),
        params[1].clamp(min=EPS),
    )


def _hurdle_poisson_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `HurdlePoisson` from ``[zero_prob, rate]``."""
    return HurdlePoisson(
        params[0].clamp(EPS, 1.0 - EPS),
        params[1].clamp(min=EPS),
    )


def _zero_one_inflated_beta_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `ZeroOneInflatedBeta` from ``[mu, phi, zoi, coi]``.

    Matches the formula-frontend emit order
    ``observe y <- ZeroOneInflatedBeta(mu, phi, zoi, coi)``.
    """
    return ZeroOneInflatedBeta(
        params[0].clamp(EPS, 1.0 - EPS),
        params[1].clamp(min=EPS),
        params[2].clamp(EPS, 1.0 - EPS),
        params[3].clamp(EPS, 1.0 - EPS),
    )


def _mixture_normal_builder(
    params: list[torch.Tensor],
) -> D.Distribution:
    """Build `MixtureNormal` from ``[weights, loc, scale]``.

    Each parameter is a per-row vector of length ``K`` (number of
    mixture components). The DSL surfaces this as:

        observe y : Resp <- MixtureNormal(weights, locs, scales)

    with `weights`, `locs`, `scales` declared as ``K``-dim variables
    (e.g. sampled per-row vectors or host-supplied data tensors).
    """
    return MixtureNormal(
        params[0],
        params[1],
        params[2].clamp(min=EPS),
    )


def _ordered_logistic_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build OrderedLogistic from ``[predictor, cutpoints]``.

    ``predictor`` is shape ``(batch,)``; ``cutpoints`` is either
    ``(batch, K-1)`` (per-row thresholds, the ordinal-mixed-model
    case) or a length-``K-1`` vector broadcast over the batch.

    Cutpoints are not sorted here; the caller (typically a learned
    monotonic transform or a fixed list) is responsible for ordering.
    Unsorted cutpoints produce negative differences and the
    distribution's log-prob clamp will return ``log(0)`` for the
    affected rows, surfacing the bug at training time.
    """
    return OrderedLogistic(params[0], params[1])


def _ordered_probit_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build OrderedProbit from ``[eta, cutpoints]``.

    Parameters
    ----------
    params : list[torch.Tensor]
        The predictor, one value per row, and the sorted cutpoints, shared
        ``(K - 1,)`` or per row ``(batch, K - 1)``.

    Returns
    -------
    torch.distributions.Distribution
        The ordered-probit distribution.
    """
    return OrderedProbit(params[0], params[1])


def _inverse_wishart_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build InverseWishart from ``[df, scale_tril]``.

    Parameters
    ----------
    params : list[torch.Tensor]
        The degrees of freedom and the lower Cholesky factor of the scale
        of the Wishart whose inverse is drawn.

    Returns
    -------
    torch.distributions.Distribution
        The inverse-Wishart distribution.
    """
    return InverseWishart(params[0], params[1])


def _matrix_normal_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build MatrixNormal from ``[loc, row_covariance, col_covariance]``.

    Parameters
    ----------
    params : list[torch.Tensor]
        The mean matrix and the row and column covariances.

    Returns
    -------
    torch.distributions.Distribution
        The matrix-normal distribution.
    """
    return MatrixNormal(params[0], params[1], params[2])


def _categorical_logits_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build a categorical distribution from one tensor of logits."""
    if len(params) != 1:
        raise ValueError(
            f"inline Categorical(logits=...) expects one tensor; got {len(params)}"
        )
    return D.Categorical(logits=params[0])


_FAMILY_BUILDERS: dict[str, tuple[tuple[str, ...], Callable, bool]] = {
    "Normal": (("loc", "scale"), _normal_builder, False),
    "Bernoulli": (("probs",), _bernoulli_builder, True),
    "Logistic": (("loc", "scale"), _logistic_builder, False),
    "HalfStudentT": (("df", "scale"), _half_student_t_builder, False),
    "BetaBinomial": (
        ("total_count", "concentration1", "concentration0"),
        _beta_binomial_builder,
        True,
    ),
    "TruncatedNormal": (
        ("mu", "sigma", "low", "high"),
        _truncated_normal_builder,
        False,
    ),
    "LogitNormal": (("mu", "sigma"), _logitnormal_builder, False),
    "Uniform": (("low", "high"), _uniform_builder, False),
    "Beta": (("concentration1", "concentration0"), _beta_builder, False),
    "Exponential": (("rate",), _exponential_builder, False),
    "HalfCauchy": (("scale",), _halfcauchy_builder, False),
    "HalfNormal": (("scale",), _halfnormal_builder, False),
    "LogNormal": (("loc", "scale"), _lognormal_builder, False),
    "Gamma": (("concentration", "rate"), _gamma_builder, False),
    "LogGamma": (("concentration", "rate"), _log_gamma_builder, False),
    "LogitBeta": (
        ("concentration1", "concentration0"),
        _logit_beta_builder,
        False,
    ),
    "Dirichlet": (("concentration",), _dirichlet_builder, False),
    "OrderedLogistic": (
        ("predictor", "cutpoints"),
        _ordered_logistic_builder,
        True,
    ),
    "OrderedProbit": (("eta", "cutpoints"), _ordered_probit_builder, True),
    "InverseWishart": (("df", "scale_tril"), _inverse_wishart_builder, False),
    "MatrixNormal": (
        ("loc", "row_covariance", "col_covariance"),
        _matrix_normal_builder,
        False,
    ),
    "ZeroInflatedPoisson": (
        ("zero_prob", "rate"),
        _zero_inflated_poisson_builder,
        True,
    ),
    "HurdlePoisson": (
        ("zero_prob", "rate"),
        _hurdle_poisson_builder,
        True,
    ),
    "ZeroOneInflatedBeta": (
        ("mu", "phi", "zoi", "coi"),
        _zero_one_inflated_beta_builder,
        False,
    ),
    "MixtureNormal": (
        ("weights", "loc", "scale"),
        _mixture_normal_builder,
        False,
    ),
}


# Per-family per-parameter event ranks for the inline observe / draw
# surface. A rank-0 parameter is a per-row scalar (participates in
# the standard `_stack_tensors` per-row concatenation). A rank-1
# parameter is a vector whose length is a distribution property
# (cutpoints for OrderedLogistic, weights for MixtureNormal, ...).
# Vector params may arrive as either a shared 1D tensor of shape
# ``(D,)`` broadcast across the plate, or a per-row 2D tensor of
# shape ``(batch, D)``. Families not listed default to all-rank-0.
_PARAM_EVENT_RANKS: dict[str, tuple[int, ...]] = {
    "OrderedLogistic": (0, 1),  # predictor scalar, cutpoints vector
    "OrderedProbit": (0, 1),
    "Categorical": (1,),  # probabilities or logits over the categories
    "Dirichlet": (1,),  # concentration over the simplex
    "MixtureNormal": (1, 1, 1),  # per-component vectors
    "ZeroInflatedPoisson": (0, 0),
    "HurdlePoisson": (0, 0),
    "ZeroOneInflatedBeta": (0, 0, 0, 0),
}


# Operator families that take Distribution-valued / list-valued args
# rather than scalar / vector parameters. Dispatched separately by
# `make_inline_distribution`; see the compositional measure algebra
# in `quivers.continuous.measure`.
_OPERATOR_FAMILIES: frozenset[str] = frozenset(
    {
        "Mixture",
        "Restrict",
        "Truncate",  # alias of Restrict
        "Pushforward",
        "PointMass",
        "Independent",
        "Normalize",
    }
)


def _unwrap_draw_arg(arg):
    """Translate a `DrawArg` tagged variant back into the plain
    Python representation the inline machinery already understands.

    `DrawArgName` -> `str`, `DrawArgScalar` -> `float`, `DrawArgIndex`
    -> [`Indexed`][quivers.continuous.program_steps.Indexed], the step
    argument a program gathers through. `DrawArgDist` and `DrawArgList`
    round-trip unchanged; the operator dispatch above handles them.
    """
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgIndex):
        return Indexed(name=arg.name, indices=arg.indices)
    return arg


def _list_values(arg: DrawArgList) -> list[float] | list[list[float]]:
    """The numbers of a list literal, nested as the literal nests them.

    Parameters
    ----------
    arg : DrawArgList
        The literal.

    Returns
    -------
    list[float] | list[list[float]]
        Its entries.

    Raises
    ------
    TypeError
        If an entry is neither a number nor a list of numbers.
    """
    values: list = []
    for item in arg.items:
        if isinstance(item, DrawArgScalar):
            values.append(float(item.value))
        elif isinstance(item, DrawArgList):
            values.append(_list_values(item))
        else:
            raise TypeError(
                f"a list literal argument holds numbers or lists of numbers, "
                f"not a {type(item).__name__}"
            )
    return values


def _inline_argument(
    arg: DrawArg | float | str | Indexed,
) -> tuple[Literal["lit", "const", "var"], float | torch.Tensor | StepArgument]:
    """Classify one inline family argument for the compiled runtime.

    Every draw argument kind the grammar has reaches this one function:
    a number (`DrawArgScalar`) is a literal, a list literal
    (`DrawArgList`) a literal tensor every row shares, a name
    (`DrawArgName`) or a bracket-indexed name (`DrawArgIndex`) a value
    read from the program's bindings, and a keyword argument
    (`DrawArgNamed`) the argument it names. A nested distribution
    (`DrawArgDist`) fills only a parameter that takes a distribution,
    which an inline family with value parameters has none of.

    Parameters
    ----------
    arg : DrawArg | float | str | Indexed
        The argument: a tagged draw argument, or its plain form (a
        number, a name, or an `Indexed` gather).

    Returns
    -------
    tuple[Literal["lit", "const", "var"], float | torch.Tensor | StepArgument]
        The argument's kind and its literal value or step argument.

    Raises
    ------
    TypeError
        If the argument is a nested distribution, or a kind the
        compiled runtime has no reading for, named in the message.
    """
    if isinstance(arg, DrawArgNamed):
        return _inline_argument(arg.value)
    if isinstance(arg, DrawArgScalar):
        return "lit", float(arg.value)
    if isinstance(arg, int | float) and not isinstance(arg, bool):
        return "lit", float(arg)
    if isinstance(arg, DrawArgList):
        return "const", torch.tensor(_list_values(arg))
    if isinstance(arg, DrawArgName):
        return "var", arg.text
    if isinstance(arg, DrawArgIndex):
        return "var", Indexed(name=arg.name, indices=arg.indices)
    if isinstance(arg, str | Indexed):
        return "var", arg
    if isinstance(arg, DrawArgDist):
        raise TypeError(
            "a nested distribution (DrawArgDist) fills a parameter that takes a value"
        )
    raise TypeError(
        f"the compiled runtime has no reading for a {type(arg).__name__} draw argument"
    )


def _normalize_inline_args(args):
    """Map a tuple of (possibly tagged) draw args to the plain
    `str | float | DrawArgDist | DrawArgList` form. Operator
    dispatch sees the tagged compound shapes; the rest of the
    inline path sees bare strings and floats.
    """
    return tuple(_unwrap_draw_arg(a) for a in args)


def _eval_draw_arg_value(
    arg,
    variable_types,
):
    """Evaluate a draw-arg to a Python value usable inside the
    measure-algebra constructors. Scalars become floats; numeric
    `DrawArgList`s become lists of floats; `DrawArgDist`s become
    `Distribution` instances built recursively.

    Identifiers stay as strings; the operator family decides what
    to do with them (e.g. `PointMass(x)` where `x` is an identifier
    must be resolved at trace time, which the operator dispatch
    handles by raising for the current implementation).
    """
    if isinstance(arg, DrawArgNamed):
        return _eval_draw_arg_value(arg.value, variable_types)
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgList):
        return [_eval_draw_arg_value(item, variable_types) for item in arg.items]
    if isinstance(arg, DrawArgDist):
        return _build_inner_distribution(arg, variable_types)
    if isinstance(arg, (int, float)):
        return float(arg)
    if isinstance(arg, str):
        return arg
    raise TypeError(f"_eval_draw_arg_value: unexpected arg type {type(arg).__name__}")


def _build_inner_distribution(
    arg: DrawArgDist,
    variable_types,
):
    """Recursively build a `torch.distributions.Distribution` (or
    `Measure`) from a `DrawArgDist`, dispatching on the family name.
    Used by the compositional measure algebra to evaluate
    distribution-valued arguments before passing them to the outer
    family's constructor.

    Recognises every operator in `_OPERATOR_FAMILIES` plus the base
    families registered in `_FIXED_FACTORIES` and `_FAMILY_BUILDERS`
    (with all-literal args; variable-bound base families inside a
    compositional expression are deferred to the operator's runtime
    evaluation).
    """
    family = arg.family
    if family in _OPERATOR_FAMILIES:
        return _build_operator_value(family, arg.args, variable_types)
    if family in _BASE_DISTRIBUTION_BUILDERS:
        bound_args, _ = _bind_inline_arguments(family, arg.args)
        evaluated_args = [_eval_draw_arg_value(a, variable_types) for a in bound_args]
        if any(isinstance(v, str) for v in evaluated_args):
            raise ValueError(
                f"Inline distribution {family}({arg.args}): "
                "variable-bound base families inside a compositional "
                "expression require the operator's runtime evaluation; "
                "use the standalone form or a let-binding."
            )
        return _BASE_DISTRIBUTION_BUILDERS[family](evaluated_args)
    raise ValueError(
        f"unknown inline family {family!r} in compositional "
        f"expression; operators: {sorted(_OPERATOR_FAMILIES)}; bases: "
        f"{sorted(_BASE_DISTRIBUTION_BUILDERS)}"
    )


# Closed-form constructors for base distributions used inside
# compositional expressions. Each takes a list of evaluated args
# (floats, after recursing through `_eval_draw_arg_value`) and
# returns a `torch.distributions.Distribution` instance. Only the
# families that compose naturally with the operator algebra are
# listed; user-supplied families can be registered here at runtime.
_BASE_DISTRIBUTION_BUILDERS: dict = {
    "Normal": lambda a: D.Normal(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Cauchy": lambda a: D.Cauchy(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Laplace": lambda a: D.Laplace(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "StudentT": lambda a: D.StudentT(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]) if len(a) > 1 else torch.tensor(0.0),
        torch.as_tensor(a[2]) if len(a) > 2 else torch.tensor(1.0),
    ),
    "Beta": lambda a: D.Beta(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Gamma": lambda a: D.Gamma(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "Exponential": lambda a: D.Exponential(
        torch.as_tensor(a[0]),
    ),
    "LogGamma": lambda a: LogGamma(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
        validate_args=True,
    ),
    "LogitBeta": lambda a: LogitBeta(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
        validate_args=True,
    ),
    "Poisson": lambda a: D.Poisson(torch.as_tensor(a[0])),
    "Bernoulli": lambda a: D.Bernoulli(probs=torch.as_tensor(a[0])),
    "Uniform": lambda a: D.Uniform(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
    "HalfNormal": lambda a: D.HalfNormal(torch.as_tensor(a[0])),
    "HalfCauchy": lambda a: D.HalfCauchy(torch.as_tensor(a[0])),
    "LogNormal": lambda a: D.LogNormal(
        torch.as_tensor(a[0]),
        torch.as_tensor(a[1]),
    ),
}


def _build_operator_value(
    family: str,
    args: tuple,
    variable_types,
    *,
    bound: bool = False,
    parameter_names: tuple[str, ...] | None = None,
):
    """Evaluate an operator-family call inside a compositional
    expression. Returns a `Distribution` / `Measure` value usable as
    a parameter to an enclosing operator.
    """
    if bound:
        values = args
        names = parameter_names
    else:
        values, names = _bind_inline_arguments(family, args)
    evaluated = [_eval_draw_arg_value(a, variable_types) for a in values]
    if family == "PointMass":
        if len(evaluated) != 1:
            raise ValueError(f"PointMass: expected 1 arg, got {len(evaluated)}")
        return _MeasurePointMass(torch.as_tensor(evaluated[0]))
    if family in ("Restrict", "Truncate"):
        if len(evaluated) < 2 or len(evaluated) > 3:
            raise ValueError(
                f"{family}: expected (base[, low, high]), got {len(evaluated)} args"
            )
        base = evaluated[0]
        if not hasattr(base, "log_prob"):
            raise ValueError(
                f"{family}: first arg must be a Distribution, got {type(base).__name__}"
            )
        named = dict(zip(names or (), evaluated, strict=False))
        low = named.get("low")
        high = named.get("high")
        low_t = torch.as_tensor(low) if low is not None else None
        high_t = torch.as_tensor(high) if high is not None else None
        return _MeasureRestrict(base, low=low_t, high=high_t)
    if family == "Pushforward":
        if len(evaluated) != 2:
            raise ValueError(
                f"Pushforward: expected (base, bijector), got {len(evaluated)} args"
            )
        base, bij = evaluated
        if isinstance(bij, str):
            bij_cls = BIJECTORS_BY_NAME.get(bij)
            if bij_cls is None:
                raise ValueError(
                    f"Pushforward: unknown bijector {bij!r}; "
                    f"register or use the named operators."
                )
            bij = bij_cls()
        if not isinstance(bij, Bijector):
            raise ValueError(
                f"Pushforward: second arg must be a Bijector, got {type(bij).__name__}"
            )
        return _MeasurePushforward(base, bij)
    if family == "Mixture":
        if len(evaluated) != 2:
            raise ValueError(
                f"Mixture: expected (weights, components), got {len(evaluated)} args"
            )
        weights_raw, components_raw = evaluated
        weights = torch.as_tensor(weights_raw, dtype=torch.float32)
        if not isinstance(components_raw, list):
            raise ValueError("Mixture: components must be a list literal")
        return _MeasureMixture(weights, components_raw)
    if family == "Independent":
        if len(evaluated) != 2:
            raise ValueError(
                "Independent: expected (base, reinterpreted_batch_ndims), "
                f"got {len(evaluated)} args"
            )
        base, n = evaluated
        return _MeasureIndependent(base, int(n))
    if family == "Normalize":
        if len(evaluated) != 1:
            raise ValueError(f"Normalize: expected 1 arg, got {len(evaluated)}")
        return _MeasureNormalize(evaluated[0])
    raise ValueError(f"unknown operator family {family!r}")


def _make_operator_distribution(
    family: str,
    args,
    codomain: AnySpace,
    variable_types,
    parameter_names: tuple[str, ...] | None,
):
    """Build a `FixedDistribution` wrapper around a compositional
    measure expression. The expression has no free QVR variables;
    all arguments are evaluated at compile time to a closed-form
    `Distribution` / `Measure` instance.
    """
    fake_dist = _build_operator_value(
        family,
        tuple(args),
        variable_types,
        bound=True,
        parameter_names=parameter_names,
    )
    support = getattr(fake_dist, "support", _constraints.real)
    discrete = isinstance(support, type(_constraints.nonnegative_integer))

    def builder(batch: int, device: torch.device):
        return fake_dist

    morph = FixedDistribution(
        codomain,
        builder,
        discrete=discrete,
        support=support,
    )
    return (morph, None)


def _param_row_width(vtype: AnySpace, event_rank: int) -> int:
    """Width one row of ``vtype`` occupies in the stacked parameters.

    Rank-zero Euclidean parameters use the space's per-row width.
    Higher-rank parameters use its flat dimension. Other spaces default
    to one column.
    """
    if event_rank == 0 and isinstance(vtype, Euclidean):
        return vtype.row_width
    return int(getattr(vtype, "dim", 1))


def _gathered_width(arg: Indexed, extents: tuple[int, ...]) -> int:
    """The width of one gathered element of a value of known extents.

    Gathering ``mu`` at ``k`` index bindings consumes its ``k``
    outermost axes, so each gathered element is the value left over the
    rest: one entry of a vector, one row of a matrix.

    Parameters
    ----------
    arg : Indexed
        The gather.
    extents : tuple[int, ...]
        The checked extents of the value gathered from.

    Returns
    -------
    int
        The number of entries of one gathered element.

    Raises
    ------
    ValueError
        If the gather indexes more axes than the value has.
    """
    if len(arg.indices) > len(extents):
        raise ValueError(
            f"{arg.name}[{', '.join(arg.indices)}] indexes {len(arg.indices)} "
            f"axes of a value with {len(extents)}"
        )
    return math.prod(extents[len(arg.indices) :])


def make_inline_distribution(
    family: str,
    args: tuple,
    codomain: AnySpace,
    variable_types: dict[str, AnySpace] | None = None,
    value_extents: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[ContinuousMorphism, tuple[StepArgument, ...] | None]:
    """Create an inline distribution from family name and mixed args.

    Handles any combination of literal and variable arguments for any
    registered distribution family. The general mechanism:

    1. All literals → ``FixedDistribution`` (no variable input)
    2. Any variables → ``MixedInlineDistribution`` with a param_spec
       that records which positions are literals vs. variable slices

    Parameters
    ----------
    family : str
        Distribution family name.
    args : tuple of str | float
        Arguments from the DSL. Strings are variable names,
        floats are literal values.
    codomain : AnySpace
        The output space for the distribution.
    variable_types : dict or None
        Mapping of variable names to their space types (for domain
        construction of direct distributions).
    value_extents : Mapping[str, tuple[int, ...]] or None
        The checked extents of the values the program binds, by name. A
        gather ``mu[z]`` reads, per index, the value left after its
        gathered axes: one entry of a vector, or one row of a matrix.

    Returns
    -------
    tuple of (ContinuousMorphism, tuple[str, ...] | None)
        The inline distribution morphism, and the variable names
        to pass as step input (None = use program input).

    Raises
    ------
    ValueError
        If a gather indexes more axes than its value's checked extents
        have.
    """
    args, selected_parameters = _bind_inline_arguments(family, args)
    if family in _OPERATOR_FAMILIES:
        return _make_operator_distribution(
            family,
            args,
            codomain,
            variable_types,
            selected_parameters,
        )
    args = _normalize_inline_args(args)
    # An indexed reference counts as a variable reference for the
    # purposes of picking the fixed vs mixed factory dispatch, alongside
    # bare identifier strings.
    var_names = [a for a in args if isinstance(a, str | Indexed)]
    # A list literal filling a parameter of a family whose fixed factory
    # takes numbers alone reads as a literal tensor of the general path
    # below, as it does beside a variable argument.
    literal_tensor = any(isinstance(a, DrawArgList) for a in args) and (
        family not in _VECTOR_PARAM_FAMILIES and family != "Categorical"
    )
    fixed = not var_names and not (literal_tensor and family in _FAMILY_BUILDERS)
    if fixed:
        if family == "Categorical":
            parameterization = (selected_parameters or ("probs",))[0]
            if not isinstance(args[0], DrawArgList):
                values = _eval_draw_arg_value(args[0], variable_types)
                return (
                    make_fixed_categorical(
                        values, codomain, parameterization=parameterization
                    ),
                    None,
                )
            listed = _list_values(args[0])
            weights = [entry for entry in listed if isinstance(entry, float)]
            if len(weights) != len(listed):
                raise ValueError(
                    "inline Categorical takes one vector of class weights, "
                    "not a nested list"
                )
            return (
                make_fixed_categorical(
                    weights, codomain, parameterization=parameterization
                ),
                None,
            )
        if family in _FIXED_FACTORIES:
            factory_parameters, factory = _FIXED_FACTORIES[family]
            if (
                selected_parameters is not None
                and selected_parameters != factory_parameters
            ):
                return (
                    _make_fixed_selected_parameters(
                        family,
                        args,
                        selected_parameters,
                        codomain,
                    ),
                    None,
                )
            # Vector-parameter families take a single ``list[float]``
            # / ``tuple[float, ...]`` argument rather than splatting
            # the literals. The parser surfaces ``Dirichlet([1, 2, 3])``
            # as three positional float args (the grammar's draw-arg
            # list flattens any bracket-bounded numeric sequence), so
            # we re-bundle here when the factory's documented contract
            # is a single vector argument.
            if family in _VECTOR_PARAM_FAMILIES:
                value = (
                    _list_values(args[0])
                    if isinstance(args[0], DrawArgList)
                    else _eval_draw_arg_value(args[0], variable_types)
                )
                morph = factory(value, codomain)
            else:
                all_floats = [float(a) for a in args]
                morph = factory(*all_floats, codomain)
            return (morph, None)
        if family not in _FAMILY_BUILDERS:
            raise ValueError(f"no fixed factory for inline family {family!r}")
    if family not in _FAMILY_BUILDERS:
        raise ValueError(
            f"no builder for inline family {family!r} with variable arguments"
        )
    param_names, dist_builder, discrete = _FAMILY_BUILDERS[family]
    if family == "Categorical" and selected_parameters == ("logits",):
        param_names = ("logits",)
        dist_builder = _categorical_logits_builder
    elif selected_parameters is not None and selected_parameters != param_names:
        spec = FAMILY_REGISTRY[family]
        source_parameters = selected_parameters

        def _selected_builder(
            params: list[torch.Tensor],
            _spec: FamilySpec = spec,
            _parameters: tuple[str, ...] = source_parameters,
        ) -> D.Distribution:
            return _spec.dist_class(
                **{
                    name: _selected_clamp(_spec, name, value)
                    for name, value in zip(_parameters, params, strict=True)
                }
            )

        param_names = selected_parameters
        dist_builder = _ALTERNATIVE_BUILDERS.get(
            (family, selected_parameters), _selected_builder
        )
    if len(args) != len(param_names):
        raise ValueError(
            f"inline {family} expects {len(param_names)} args ({', '.join(param_names)}), got {len(args)}"
        )
    fam_event_ranks = _PARAM_EVENT_RANKS.get(family) or _registry_event_ranks(
        family, param_names
    )
    param_spec: list[tuple[str, int | float | torch.Tensor | None]] = []
    param_shapes: list[tuple[int, ...] | None] = []
    var_name_order: list[StepArgument] = []
    total_var_dim = 0
    for i, arg in enumerate(args):
        kind, value = _inline_argument(arg)
        param_shapes.append(None)
        if kind == "lit" or kind == "const":
            assert isinstance(value, float | torch.Tensor)
            param_spec.append((kind, value))
        else:
            assert isinstance(value, str | Indexed)
            arg = value
            # A parameter's width comes from its binding's type; without
            # one it is left unresolved, for the stacked width to settle.
            var_dim: int | None = None
            lookup_name = arg.name if isinstance(arg, Indexed) else arg
            event_rank = fam_event_ranks[i]
            checked = (
                value_extents.get(lookup_name) if value_extents is not None else None
            )
            if (
                isinstance(arg, Indexed)
                and value_extents is not None
                and lookup_name in value_extents
            ):
                var_dim = _gathered_width(arg, value_extents[lookup_name])
            elif (
                not isinstance(arg, Indexed)
                and event_rank >= 1
                and checked is not None
                and len(checked) >= event_rank
            ):
                # A vector or matrix parameter's own axes are the trailing
                # ones of its checked value; any leading axis is a plate's.
                own = tuple(checked[len(checked) - event_rank :])
                var_dim = math.prod(own)
                if event_rank >= 2:
                    param_shapes[-1] = own
            elif variable_types and lookup_name in variable_types:
                vtype = variable_types[lookup_name]
                if isinstance(arg, Indexed) and isinstance(vtype, Euclidean):
                    # With no checked extents, a gather from a plate
                    # selects one of its declared rows; from a value
                    # with no plate rows its width is left for the
                    # stacked width to settle.
                    if vtype.plate_rows is not None:
                        var_dim = vtype.row_width
                else:
                    var_dim = _param_row_width(vtype, fam_event_ranks[i])
            if event_rank >= 2 and param_shapes[-1] is None:
                raise UnsizedParameterError(
                    f"inline {family} reads its {param_names[i]!r} from "
                    f"{lookup_name!r}, a matrix whose shape the checked program "
                    "leaves to the data; the compiled runtime sizes every "
                    "parameter when it builds the program, so fix the draw's "
                    "event axes with an annotation or ``[over=...]``, or return "
                    "the draw from a program whose codomain fixes them"
                )
            param_spec.append(("var", var_dim))
            var_name_order.append(arg)
            total_var_dim += 1 if var_dim is None else var_dim
    if family == "MixtureNormal":
        # QIEC can currently retain the component width for an indexed draw
        # while conservatively typing a derived vector-valued ``let`` as
        # width one. MixtureNormal's three vector parameters necessarily
        # share one component axis, so a known width on any slot determines
        # the unresolved derived slots without guessing at runtime after the
        # arguments have been concatenated.
        known_component_dims = {
            int(value)
            for (kind, value), rank in zip(param_spec, fam_event_ranks, strict=True)
            if kind == "var" and rank >= 1 and value is not None and int(value) > 1
        }
        if len(known_component_dims) == 1:
            component_dim = known_component_dims.pop()
            param_spec = [
                (kind, component_dim)
                if kind == "var" and rank >= 1 and (value is None or int(value) == 1)
                else (kind, value)
                for (kind, value), rank in zip(param_spec, fam_event_ranks, strict=True)
            ]
            total_var_dim = sum(
                1 if value is None else int(value)
                for kind, value in param_spec
                if kind == "var"
            )
    if total_var_dim == 0:
        domain = Euclidean(name="_inline_domain", dim=1)
    elif len(var_name_order) == 1 and variable_types:
        vtype = _argument_type(var_name_order[0], variable_types)
        domain = (
            vtype
            if vtype is not None
            else Euclidean(name="_inline_domain", dim=total_var_dim)
        )
    else:
        domain = _infer_domain(var_name_order, variable_types)
    fam_support: _constraints.Constraint = _FAMILY_SUPPORTS.get(
        family, _constraints.real
    )
    # TruncatedNormal / Uniform encode bounded supports via two literal
    # arguments; if both bounds are literal we can specialize the
    # constraint to the matching `interval`. Otherwise we fall
    # back to ``real`` (the closest correct guide-side approximation).
    if family in ("TruncatedNormal", "Uniform"):
        lit_args: list[float] = []
        for kind, value in param_spec:
            if kind == "lit" and value is not None:
                lit_args.append(float(value))
        if family == "Uniform" and len(lit_args) == 2:
            fam_support = _constraints.interval(lit_args[0], lit_args[1])
        elif family == "TruncatedNormal" and len(lit_args) >= 2:
            fam_support = _constraints.interval(lit_args[-2], lit_args[-1])
    # Dimension-dependent families need the codomain's matrix / vector
    # size baked into the builder closure. The registered
    # `_FAMILY_BUILDERS` entry carries a dim-agnostic builder; here we
    # replace it with a closure that captures the codomain dim so the
    # resulting torch distribution has the right event_shape at
    # construction.
    if family in _DIM_DEPENDENT_FAMILIES:
        dist_builder = _dim_dependent_builder(family, codomain)
    morph = MixedInlineDistribution(
        domain,
        codomain,
        param_spec,
        dist_builder,
        discrete,
        support=fam_support,
        param_event_ranks=fam_event_ranks,
        param_shapes=tuple(param_shapes),
    )
    return (morph, tuple(var_name_order) if var_name_order else None)


def _registry_event_ranks(family: str, parameters: tuple[str, ...]) -> tuple[int, ...]:
    """The rank of one value of each parameter, as the family registry records it.

    Parameters
    ----------
    family : str
        The family's source name.
    parameters : tuple[str, ...]
        The parameter names the call supplies, in order, as the source
        spells them.

    Returns
    -------
    tuple[int, ...]
        One rank per parameter: the registry's, through the family's
        surface aliases, and zero for a parameter the registry does not
        record.
    """
    record = FAMILIES.get(FAMILY_ALIASES.get(family, family))
    aliases = PARAMETER_ALIASES.get(family, {})
    ranks: list[int] = []
    for name in parameters:
        semantic = aliases.get(name, name)
        ranks.append(
            record.parameter(semantic).rank
            if record is not None and semantic in record.parameter_names
            else 0
        )
    return tuple(ranks)


def _make_fixed_selected_parameters(
    family: str,
    args: tuple,
    parameters: tuple[str, ...],
    codomain: AnySpace,
) -> FixedDistribution:
    """Build a literal family through the parameterization named in QVR."""
    spec = FAMILY_REGISTRY[family]
    evaluated = [_eval_draw_arg_value(argument, None) for argument in args]
    d = int(getattr(codomain, "dim", 1))

    def builder(batch: int, device: torch.device) -> D.Distribution:
        values: dict[str, torch.Tensor] = {}
        for name, value in zip(parameters, evaluated, strict=True):
            if isinstance(value, (list, tuple)):
                literal = torch.as_tensor(value, device=device)
                tensor = literal.expand(batch, *literal.shape)
            else:
                tensor = torch.full((batch, d), float(value), device=device)
            values[name] = _selected_clamp(spec, name, tensor)
        alternative = _ALTERNATIVE_BUILDERS.get((family, parameters))
        if alternative is not None:
            return alternative([values[name] for name in parameters])
        return spec.dist_class(**values)

    return FixedDistribution(
        codomain,
        builder,
        discrete=spec.discrete,
        support=spec.support,
    )


class _LogitGeometric(D.Geometric):
    """A geometric distribution scored from its logits in log space.

    `torch.distributions.Geometric` scores through its probability,
    whose complement rounds to zero once the logit passes about 17 in
    single precision. With ``log p = -softplus(-l)`` and
    ``log (1 - p) = -softplus(l)`` the count of failures ``k`` scores
    ``k log (1 - p) + log p`` at every finite logit.

    Parameters
    ----------
    logits : torch.Tensor
        The log-odds of a success.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__(logits=logits)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """The log mass at a count of failures.

        Parameters
        ----------
        value : torch.Tensor
            The counts.

        Returns
        -------
        torch.Tensor
            The log masses.
        """
        if self._validate_args:
            self._validate_sample(value)
        value, logits = torch.broadcast_tensors(value, self.logits)
        failures = value * -torch.nn.functional.softplus(logits)
        return failures - torch.nn.functional.softplus(-logits)


def _logit_geometric_builder(params: list[torch.Tensor]) -> D.Distribution:
    """Build a geometric distribution from ``[logits]``."""
    return _LogitGeometric(params[0])


#: Builders for an alternative parameterization whose `torch.distributions`
#: class loses precision there, keyed by family and the parameters the
#: source names.
_ALTERNATIVE_BUILDERS: dict[
    tuple[str, tuple[str, ...]], Callable[[list[torch.Tensor]], D.Distribution]
] = {("Geometric", ("logits",)): _logit_geometric_builder}


def _selected_clamp(spec: FamilySpec, name: str, value: torch.Tensor) -> torch.Tensor:
    """Keep a parameter the source names inside its family's constraint.

    A probability is held off zero and one, a parameter the family's
    specification declares takes its own inline clamp, and an alternative
    the specification does not declare (logits, a covariance or a
    precision matrix) is passed as written: logits range over the reals,
    and a matrix's own validation reports one that is not positive
    definite.

    Parameters
    ----------
    spec : FamilySpec
        The family's specification.
    name : str
        The parameter the source names.
    value : torch.Tensor
        Its value.

    Returns
    -------
    torch.Tensor
        The value the family is built with.
    """
    if name == "probs":
        return value.clamp(min=EPS, max=1.0 - EPS)
    declared = {param.name: param for param in spec.params}
    if name in declared:
        return declared[name].inline_clamp(value)
    return value


def _bind_inline_arguments(
    family: str,
    args: tuple,
) -> tuple[tuple, tuple[str, ...] | None]:
    """Validate and order a keyword-only family construction.

    Argument names are structural `DrawArgNamed` wrappers. The complete set
    must match one declared family schema; returned values follow that
    schema's order for the positional runtime builders.
    """
    if not args:
        return args, None
    bound = bind_family_arguments(family, args, get_inline_param_names(family) or ())
    return tuple(value for _, value in bound), tuple(name for name, _ in bound)


def _argument_type(
    arg: StepArgument, variable_types: dict[str, AnySpace] | None
) -> AnySpace | None:
    """The known type of a bound argument.

    Parameters
    ----------
    arg : StepArgument
        A binding's name, or an indexed gather of one.
    variable_types : dict or None
        Known variable types.

    Returns
    -------
    AnySpace or None
        The binding's type, or ``None`` for an unknown binding and for a
        gather, whose rows the base binding's type does not describe.
    """
    if variable_types is None or isinstance(arg, Indexed):
        return None
    return variable_types.get(arg)


def _infer_domain(
    var_names: list[StepArgument], variable_types: dict[str, AnySpace] | None
) -> AnySpace:
    """Infer a domain space from variable types.

    Parameters
    ----------
    var_names : list[StepArgument]
        The bound arguments used as input.
    variable_types : dict or None
        Known variable types.

    Returns
    -------
    AnySpace
        The inferred domain.
    """
    if variable_types is None or not var_names:
        return Euclidean(name="_inline_domain", dim=len(var_names))
    if len(var_names) == 1:
        vtype = _argument_type(var_names[0], variable_types)
        if vtype is not None:
            return vtype
        return Euclidean(name="_inline_domain", dim=1)
    from quivers.core.objects import ProductSet
    from quivers.continuous.spaces import ContinuousSpace, ProductSpace

    components = []
    for vn in var_names:
        vtype = _argument_type(vn, variable_types)
        if isinstance(vtype, Euclidean) and vtype.plate_rows is not None:
            # A plate-row parameter contributes one row of its plate to
            # each row of the stacked input, not the whole plate.
            label = vn.name if isinstance(vn, Indexed) else vn
            components.append(Euclidean(name=f"_inline_{label}", dim=vtype.row_width))
        elif vtype is not None:
            components.append(vtype)
        else:
            label = vn.name if isinstance(vn, Indexed) else vn
            components.append(Euclidean(name=f"_inline_{label}", dim=1))
    if any((isinstance(c, ContinuousSpace) for c in components)):
        return ProductSpace(components=tuple(components))
    return ProductSet(components=tuple(components))


# ---------------------------------------------------------------------------
# Auto-generated inline support from `FAMILY_REGISTRY`.
#
# For every registered family with ``output_kind == "independent"``
# that isn't already in the hand-written dicts, build a generic
# `FixedDistribution` factory and a matching mixed-mode
# builder.  This is the architectural seam that closes the gap
# between the conditional path and the inline path: a family
# declared once via `register_family`
# automatically becomes usable in DSL ``F(args)`` syntax.
# ---------------------------------------------------------------------------


def _build_generic_fixed_factory(spec: FamilySpec) -> Callable:
    """Return a fixed-factory callable matching the
    ``make_fixed_X(literal_args..., codomain)`` calling convention.

    The factory broadcasts each scalar literal to ``(batch, d)``
    where ``d = getattr(codomain, "dim", 1)`` and clamps the value
    by the per-parameter inline clamp before passing it to the
    underlying torch distribution.  Discrete families have their
    output cast to ``long`` by `FixedDistribution`.
    """

    def factory(*all_args) -> FixedDistribution:
        # Match the existing factory convention from
        # ``make_inline_distribution``: positional literals followed
        # by the codomain as the final positional argument.
        if len(all_args) != len(spec.params) + 1:
            raise ValueError(
                f"inline {spec.name} expects "
                f"{len(spec.params)} literal args "
                f"({', '.join(spec.param_names)}) plus codomain; "
                f"got {len(all_args)} args"
            )
        *literal_args, codomain = all_args
        d = getattr(codomain, "dim", 1)
        clamped_floats = []
        for param, val in zip(spec.params, literal_args):
            f = float(val)
            clamped_floats.append(float(param.inline_clamp(torch.tensor(f)).item()))

        def builder(batch: int, device: torch.device) -> D.Distribution:
            kwargs = {}
            for param, val in zip(spec.params, clamped_floats):
                kwargs[param.name] = torch.full((batch, d), val, device=device)
            return spec.dist_class(**kwargs)

        return FixedDistribution(
            codomain,
            builder,
            discrete=spec.discrete,
            support=spec.support,
        )

    return factory


def _build_generic_mixed_builder(spec: FamilySpec) -> Callable:
    """Return a mixed-mode builder accepting a list of resolved
    parameter tensors and returning a torch distribution.

    Each tensor is passed through the per-parameter inline clamp
    before construction.
    """

    def builder(params: list[torch.Tensor]) -> D.Distribution:
        if len(params) != len(spec.params):
            raise ValueError(
                f"inline {spec.name} mixed builder expects "
                f"{len(spec.params)} tensors; got {len(params)}"
            )
        kwargs = {}
        for param, t in zip(spec.params, params):
            kwargs[param.name] = param.inline_clamp(t)
        return spec.dist_class(**kwargs)

    return builder


def _auto_register_inline(spec: FamilySpec) -> None:
    """Populate the inline dicts for ``spec`` unless a hand-written
    entry is already present.

    Output-kind handling:

    * ``independent`` families register both a fixed factory (all
      literal scalar args broadcast to ``(batch, dim)``) and a mixed
      builder.
    * ``categorical`` / ``vector`` / ``mvn`` / ``matrix`` families
      register only the mixed builder. Their parameters are vector-
      or matrix-shaped, so the literal-broadcast path doesn't make
      sense; calling ``F(args)`` with at least one variable argument
      goes through the mixed path, which forwards the resolved
      parameter tensor straight to the underlying torch distribution
      constructor.
    """
    if spec.output_kind not in {
        "independent",
        "categorical",
        "vector",
        "mvn",
        "matrix",
    }:
        return
    # A hand-written entry takes precedence in its own table: a family
    # with a hand-written fixed factory still gets the generic mixed
    # builder, and the other way round.
    if spec.output_kind == "independent" and spec.name not in _FIXED_FACTORIES:
        factory = _build_generic_fixed_factory(spec)
        _FIXED_FACTORIES[spec.name] = (spec.param_names, factory)
    if spec.name not in _FAMILY_BUILDERS:
        builder = _build_generic_mixed_builder(spec)
        _FAMILY_BUILDERS[spec.name] = (spec.param_names, builder, spec.discrete)
        _FAMILY_SUPPORTS.setdefault(spec.name, spec.support)


# Ensure [`quivers.continuous.families`][quivers.continuous.families] has registered every
# family before we walk the registry. Imported here rather than at
# module top to break the import cycle (families.py imports inline
# indirectly via the DSL compiler).
from quivers.continuous import families as _families  # noqa: E402, F401

for _spec in FAMILY_REGISTRY.values():
    _auto_register_inline(_spec)


def reload_inline_registry() -> None:
    """Refresh the inline dicts from
    `quivers.continuous.family_spec.FAMILY_REGISTRY`. Useful
    when new families are registered after `inline` has
    imported (test plugins, downstream extensions)."""
    for spec in FAMILY_REGISTRY.values():
        _auto_register_inline(spec)


__all__ = [
    "FixedDistribution",
    "MixedInlineDistribution",
    "get_inline_param_names",
    "get_inline_parameterizations",
    "make_fixed_bernoulli",
    "make_fixed_beta",
    "make_fixed_categorical",
    "make_fixed_dirichlet",
    "make_fixed_exponential",
    "make_fixed_gamma",
    "make_fixed_halfcauchy",
    "make_fixed_halfnormal",
    "make_fixed_lkj_cholesky",
    "make_fixed_log_gamma",
    "make_fixed_logit_beta",
    "make_fixed_logitnormal",
    "make_fixed_lognormal",
    "make_fixed_normal",
    "make_fixed_truncated_normal",
    "make_fixed_uniform",
    "make_inline_distribution",
    "reload_inline_registry",
]
