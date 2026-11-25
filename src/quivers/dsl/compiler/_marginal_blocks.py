"""Grouped ``marginalize`` blocks: scores over units, and response laws.

A grouped block draws one class per unit of its grouping plate and sums
the class out. Its score is a sum of one log weight per unit, and every
row of a response belongs to the unit its index map names. The two
records here hold that structure as data, so that the score, the weights
of the units, the row maps, and the law of each response all read one
definition of the block.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from typing import Literal, Protocol, cast

import didactic.api as dx
import torch
from torch.distributions import Distribution, constraints as _constraints

from quivers._observation import ObservationEvent
from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous._closed_probability import _count_log
from quivers.continuous._observation_events import event_log_prob
from quivers.continuous.inline import _discrete_sample, _discrete_value
from quivers.continuous.marginal import GroupLevel, draw_classes
from quivers.continuous.measure import Independent, Mixture, truncated_law
from quivers.continuous.morphisms import BoundEvent, ContinuousMorphism
from quivers.continuous.plate import (
    VectorisedObserve,
    _flatten_product_indices,
    marginalize_grouped,
)
from quivers.continuous.program_steps import (
    Environment,
    Indexed,
    PlateMembers,
    ResponseLaw,
    RunConditions,
    StepArgument,
    StepFunction,
    UnitFactorization,
    UnitRows,
    reading,
    reads_of,
)
from quivers.continuous.param_transforms import _validate_parameter
from quivers.continuous.programs import _lookup_arg
from quivers.continuous.unit_subsets import unit_coefficients, units_subsampled
from quivers.dsl.compiler._prelude import CompileError
from quivers.effects._observation_events import RecordedObservation, resolve_event
from quivers.effects._observation_weights import WeightedObservation, checked_weights

type Reduction = Literal["logsumexp", "sum", "mean"]
"""How a block aggregates its latent's values."""


def _binary_log_prior(
    weight: torch.Tensor, parameterization: str | None
) -> torch.Tensor:
    """The log weights of the atoms 0 and 1 of a Bernoulli-family prior.

    From ``logits`` the weights are ``log sigmoid(-l)`` and
    ``log sigmoid(l)``, computed in log space, so both stay finite and
    accurate where the probability ``sigmoid(l)`` rounds to zero or one
    in the runtime's precision. From ``probs`` they are ``log1p(-p)`` and
    ``log p``, retaining zero probability as negative infinity.

    Parameters
    ----------
    weight : torch.Tensor
        The prior's probability, or its logits.
    parameterization : str | None
        ``"logits"`` when `weight` holds logits; anything else reads it
        as a probability.

    Returns
    -------
    torch.Tensor
        The log weights, stacked on a new trailing axis in atom order.
    """
    if parameterization == "logits":
        return torch.stack(
            [
                torch.nn.functional.logsigmoid(-weight),
                torch.nn.functional.logsigmoid(weight),
            ],
            dim=-1,
        )
    p = torch.distributions.Bernoulli(probs=weight, validate_args=True).probs
    one = torch.ones_like(p)
    return torch.stack([_count_log(one, 1 - p), _count_log(one, p)], dim=-1)


def _categorical_log_prior(
    weight: torch.Tensor, parameterization: str | None
) -> torch.Tensor:
    """Normalized log weights of a categorical prior.

    Parameters
    ----------
    weight : torch.Tensor
        Nonnegative weights or logits, with categories on the trailing axis.
    parameterization : str | None
        ``"logits"`` for logits, otherwise probability weights.

    Returns
    -------
    torch.Tensor
        Normalized log weights, retaining impossible atoms as negative infinity.

    Raises
    ------
    ValueError
        If the weights do not define a categorical distribution.
    """
    if parameterization == "logits":
        _validate_parameter(weight, _constraints.real)
        return torch.log_softmax(weight, dim=-1)
    _validate_parameter(weight, _constraints.nonnegative)
    probs = torch.distributions.Categorical(probs=weight, validate_args=True).probs
    return _count_log(torch.ones_like(probs), probs)


def _index_map_tensor(env: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """The index map an observation names with ``via``.

    The map sends each row to a position of the grouping plate, and is host
    data like any free covariate.

    Parameters
    ----------
    env : dict[str, torch.Tensor]
        The bindings.
    name : str
        The map.

    Returns
    -------
    torch.Tensor
        Its value.

    Raises
    ------
    CompileError
        If nothing binds the map.
    """
    idx = env.get(name)
    if idx is None:
        raise CompileError(
            f"grouped marginalize: the ``via`` index map {name!r} "
            f"is neither a bound program variable nor supplied at runtime; "
            f"pass it in the observations dict as a long tensor of "
            f"row-to-group indices",
            0,
            0,
        )
    return idx


def _project_nested_aggregates(
    aggregates: torch.Tensor,
    inner_sizes: tuple[int, ...],
    kept_axes: tuple[int, ...],
    slot: str,
) -> torch.Tensor:
    """Project an inner grouped block's per-position aggregates onto
    the enclosing block's grouping plate.

    The inner plate is a product whose factors include every factor
    of the enclosing plate; the aggregates are summed along the
    factors the enclosing plate lacks and laid out in the enclosing
    plate's factor order, flat in row-major order.

    Parameters
    ----------
    aggregates : torch.Tensor
        The inner block's aggregates, flat over its product plate, followed
        by any enclosing class axes.
    inner_sizes : tuple[int, ...]
        The cardinality of each factor of the inner plate.
    kept_axes : tuple[int, ...]
        The position in ``inner_sizes`` of each factor of the
        enclosing plate, in the enclosing plate's order.
    slot : str
        The environment slot the aggregates were read from, named in
        the error.

    Returns
    -------
    torch.Tensor
        The aggregates over the enclosing plate, preserving class axes.

    Raises
    ------
    ValueError
        If the aggregates do not have one entry per inner position.
    """
    total = 1
    for size in inner_sizes:
        total *= size
    if aggregates.dim() < 1 or aggregates.shape[0] != total:
        raise ValueError(
            f"grouped marginalize: the nested block at slot {slot!r} must "
            f"contribute one aggregate per position of its plate "
            f"{inner_sizes}; got shape {tuple(aggregates.shape)}"
        )
    extra_shape = tuple(aggregates.shape[1:])
    shaped = aggregates.reshape(*inner_sizes, *extra_shape)
    summed_axes = tuple(
        axis for axis in range(len(inner_sizes)) if axis not in kept_axes
    )
    if summed_axes:
        shaped = shaped.sum(dim=summed_axes)
        remaining = [axis for axis in range(len(inner_sizes)) if axis in kept_axes]
        order = tuple(remaining.index(axis) for axis in kept_axes)
    else:
        order = kept_axes
    order = (*order, *range(len(kept_axes), shaped.dim()))
    kept_size = 1
    for axis in kept_axes:
        kept_size *= inner_sizes[axis]
    return shaped.permute(order).reshape(kept_size, *extra_shape)


def _observation_parameter(
    env: Mapping[str, torch.Tensor],
    arguments: tuple[StepArgument, ...] | None,
    shared_vector: bool,
) -> torch.Tensor:
    """The parameter an observation family is scored at, at one atom.

    Parameters
    ----------
    env : Mapping[str, torch.Tensor]
        The bindings at the atom.
    arguments : tuple[StepArgument, ...] | None
        The observation's arguments; ``None`` reads the program input.
    shared_vector : bool
        Whether the family takes one vector-valued parameter, which a
        single shared value presents with a leading axis of one.

    Returns
    -------
    torch.Tensor
        The parameter: one argument as it is (a number as one column),
        or several stacked along a last axis.
    """
    if arguments is None:
        return env["_x_input"]
    parts = [_lookup_arg(env, argument) for argument in arguments]
    if len(parts) == 1:
        theta = parts[0]
        if theta.dim() == 0:
            # One value at this atom, shared by every row: a single
            # column, as the stacking of several parameters gives each.
            theta = theta.reshape(1)
        if shared_vector and theta.dim() == 1:
            theta = theta.unsqueeze(0)
        return theta
    common = torch.broadcast_shapes(*(part.shape for part in parts))
    theta = torch.stack([part.expand(common) for part in parts], dim=-1)
    if theta.dim() == 1:
        # Values shared by every row at this atom stack to one row of
        # their columns.
        theta = theta.unsqueeze(0)
    return theta


#: A run that evaluates every factor in full.
UNCONDITIONED = RunConditions()


def unit_positions(
    index_maps: tuple[torch.Tensor, ...], sizes: tuple[int, ...]
) -> torch.Tensor:
    """The unit each row belongs to, counted row-major over the unit axes.

    Parameters
    ----------
    index_maps : tuple[torch.Tensor, ...]
        One index per unit axis, each of shape ``(rows,)``.
    sizes : tuple[int, ...]
        The extent of each unit axis.

    Returns
    -------
    torch.Tensor
        The unit of each row, ``(rows,)``.

    Raises
    ------
    ValueError
        If an index lies outside its axis, or the maps disagree on the
        number of rows.
    """
    return _flatten_product_indices(index_maps, sizes)[0]


class _ParameterBuilder(Protocol):
    """A family that builds its distribution from resolved parameters."""

    _param_spec: list[tuple[str, float | None]]
    _discrete: bool

    def _dist_builder(self, params: list[torch.Tensor]) -> Distribution:
        """Build the family's distribution.

        Parameters
        ----------
        params : list[torch.Tensor]
            One value per parameter.

        Returns
        -------
        Distribution
            The distribution.
        """
        ...


class GroupedObservation(ValueModel):
    """An observation inside a grouped block, scored at every class.

    Parameters
    ----------
    response : str
        The observed name.
    family : ContinuousMorphism
        The response family.
    arguments : tuple[StepArgument, ...] or None
        The family's arguments; ``None`` reads the program's input.
    rows : int
        The declared extent of the observation plate.
    row_axis : str
        The object of the observation plate.
    classes : tuple[str, ...]
        The latents the likelihood has a class axis for, outermost first.
    class_sizes : tuple[int, ...]
        The extent of each class axis.
    argument_classes : tuple[tuple[str, ...], ...]
        The class axes each argument leads with.
    variable_ranks : tuple[int, ...]
        The event rank of each argument.
    group_arguments : tuple[bool, ...]
        Whether each argument is indexed by the grouping plate.
    group_sizes : tuple[int, ...]
        The extent of each axis of the grouping plate.
    index_maps : tuple[str, ...]
        The name of each axis's index map.
    weights : str or float or None
        The likelihood powers of the rows.
    event : ObservationEvent or None
        The recording roles.
    """

    response: str
    family: ContinuousMorphism = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    arguments: tuple[StepArgument, ...] | None = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    rows: int
    row_axis: str
    classes: tuple[str, ...]
    class_sizes: tuple[int, ...]
    argument_classes: tuple[tuple[str, ...], ...]
    variable_ranks: tuple[int, ...]
    group_arguments: tuple[bool, ...]
    group_sizes: tuple[int, ...]
    index_maps: tuple[str, ...]
    weights: str | float | None = None
    event: ObservationEvent | None = None

    @property
    def parameter_reads(self) -> frozenset[str]:
        """The names the response law reads.

        Returns
        -------
        frozenset[str]
            The program input, the arguments, the index maps, and the
            recording roles.
        """
        names: set[str] = {"_x_input", *self.index_maps}
        for argument in self.arguments or ():
            if isinstance(argument, Indexed):
                names.update((argument.name, *argument.indices))
            else:
                names.add(argument)
        if self.event is not None:
            names.update(self.event.names())
        return frozenset(names)

    @property
    def score_reads(self) -> frozenset[str]:
        """The names the score reads.

        Returns
        -------
        frozenset[str]
            The response and the likelihood powers, with the names the
            response law reads.
        """
        names = {self.response, *self.parameter_reads}
        if isinstance(self.weights, str):
            names.add(self.weights)
        return frozenset(names)

    @property
    def crosses(self) -> bool:
        """Whether some parameter depends on a class.

        Returns
        -------
        bool
            ``False`` when every class has the same likelihood.
        """
        return any(self.argument_classes)

    @property
    def _builder(self) -> _ParameterBuilder:
        """The family as a builder of distributions.

        Returns
        -------
        _ParameterBuilder
            The same family.
        """
        return cast(_ParameterBuilder, self.family)

    @property
    def _literal_parameters(self) -> bool:
        """Whether the family is built from resolved parameters.

        Returns
        -------
        bool
            ``True`` for an inline family.
        """
        return getattr(self.family, "_param_spec", None) is not None

    def unit_rows(self, env: Environment) -> torch.Tensor:
        """The unit of the block's grouping plate each row belongs to.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            The units, ``(rows,)``, row-major over the plate's axes.

        Raises
        ------
        CompileError
            If an index map has no value.
        ValueError
            If an index lies outside its axis.
        """
        return unit_positions(self.index_tensors(env), self.group_sizes)

    def index_tensors(self, env: Environment) -> tuple[torch.Tensor, ...]:
        """The index map of each axis of the grouping plate.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One long tensor of shape ``(rows,)`` per axis.

        Raises
        ------
        CompileError
            If an index map has no value.
        """
        maps: list[torch.Tensor] = []
        for name in self.index_maps:
            index = _index_map_tensor(env, name)
            if index.dim() == 2 and 1 in index.shape:
                # A map may carry a run axis of length one, or a column.
                index = index.reshape(-1)
            maps.append(index.to(torch.long))
        return tuple(maps)

    def _parts(
        self, env: Environment, rows: int
    ) -> tuple[list[torch.Tensor], list[bool]]:
        """The arguments' values, with group-indexed ones read at each row.

        Parameters
        ----------
        env : Environment
            The bindings.
        rows : int
            The number of rows.

        Returns
        -------
        tuple[list[torch.Tensor], list[bool]]
            The values, and whether each leads with the row axis.
        """
        if self.arguments is None:
            parts = [env["_x_input"]]
        else:
            parts = [_lookup_arg(env, argument) for argument in self.arguments]
        axes_of = self.argument_classes or ((),) * len(parts)
        ranks = self.variable_ranks or (0,) * len(parts)
        group_rank = len(self.group_sizes)
        grouped = tuple(
            declared and part.dim() >= len(axes) + group_rank + rank
            for part, axes, declared, rank in zip(
                parts,
                axes_of,
                self.group_arguments or (False,) * len(parts),
                ranks,
                strict=True,
            )
        )
        if any(grouped):
            indices = self.index_tensors(env)
            parts = [
                part[
                    (
                        *(slice(None) for _ in axes),
                        *(index.to(part.device) for index in indices),
                    )
                ].movedim(len(axes), 0)
                if pulled
                else part
                for part, axes, pulled in zip(parts, axes_of, grouped, strict=True)
            ]
        row_parts = [
            pulled or (not axes and part.dim() > rank and int(part.shape[0]) == rows)
            for part, axes, pulled, rank in zip(
                parts, axes_of, grouped, ranks, strict=True
            )
        ]
        return parts, row_parts

    def _class_parameters(
        self, parts: list[torch.Tensor], row_parts: list[bool], rows: int
    ) -> list[torch.Tensor]:
        """The family's parameters at every row and class.

        Parameters
        ----------
        parts : list[torch.Tensor]
            The arguments' values.
        row_parts : list[bool]
            Whether each leads with the row axis.
        rows : int
            The number of rows.

        Returns
        -------
        list[torch.Tensor]
            One tensor per parameter of the family, literals included,
            ``(rows, *class_sizes, *parameter_event)``.
        """
        ranks = self.variable_ranks or (0,) * len(parts)
        coordinates = not any(ranks)
        event_shape: tuple[int, ...] = ()
        aligned: list[torch.Tensor] = []
        for part, axes, is_row in zip(
            parts, self.argument_classes, row_parts, strict=True
        ):
            offset = int(is_row)
            suffix = tuple(part.shape[offset + len(axes) :])
            sizes = dict(
                zip(axes, part.shape[offset : offset + len(axes)], strict=True)
            )
            aligned.append(
                part.reshape(
                    part.shape[0] if is_row else 1,
                    *(sizes.get(name, 1) for name in self.classes),
                    *suffix,
                )
            )
            if coordinates:
                event_shape = torch.broadcast_shapes(event_shape, suffix)
        prefix = (rows, *self.class_sizes)
        resolved: list[torch.Tensor] = []
        for part in aligned:
            suffix = tuple(part.shape[len(prefix) :])
            if coordinates:
                part = part.reshape(
                    *part.shape[: len(prefix)],
                    *((1,) * (len(event_shape) - len(suffix))),
                    *suffix,
                )
                suffix = event_shape
            resolved.append(part.expand(*prefix, *suffix))
        if not self._literal_parameters:
            return resolved
        # Literal parameters share the resolved tensors' device; only the
        # declared class axes are broadcast as enumerations.
        target = (*prefix, *event_shape)
        variables = iter(resolved)
        return [
            torch.full(
                target,
                float(cast(float, literal)),
                dtype=resolved[0].dtype,
                device=resolved[0].device,
            )
            if kind == "lit"
            else next(variables)
            for kind, literal in self._builder._param_spec
        ]

    def _shared_parameters(
        self, parts: list[torch.Tensor], rows: int | None = None
    ) -> list[torch.Tensor]:
        """The family's parameters when no class changes them.

        Parameters
        ----------
        parts : list[torch.Tensor]
            The arguments' values.
        rows : int or None
            Lay a value that every row shares out along this many rows;
            ``None`` leaves the values at their own layout.

        Returns
        -------
        list[torch.Tensor]
            One tensor per parameter of the family, literals included.
        """
        ranks = self.variable_ranks or (0,) * len(parts)
        if rows is not None:
            parts = [
                part.expand(rows, *part.shape[part.dim() - rank :])
                if part.dim() <= rank or int(part.shape[0]) == 1
                else part
                for part, rank in zip(parts, ranks, strict=True)
            ]
        variables = iter(parts)
        return [
            (
                parts[0].new_tensor(float(cast(float, literal)))
                if rows is None
                else parts[0].new_full((rows,), float(cast(float, literal)))
            )
            if kind == "lit"
            else next(variables)
            for kind, literal in self._builder._param_spec
        ]

    def _roles(self, env: Environment) -> BoundEvent | None:
        """The recording roles at one run's values.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        BoundEvent or None
            The roles, or ``None`` for an observation recorded exactly.
        """
        return None if self.event is None else resolve_event(self.event, env)

    def _score(
        self,
        distribution: Distribution,
        target: torch.Tensor,
        roles: BoundEvent | None,
        rows: int,
    ) -> torch.Tensor:
        """Score a class-expanded law with its recording roles.

        Parameters
        ----------
        distribution : Distribution
            The law.
        target : torch.Tensor
            The response, broadcastable against the law.
        roles : BoundEvent or None
            The recording roles.
        rows : int
            The number of rows scored.

        Returns
        -------
        torch.Tensor
            The log densities at the law's batch layout.
        """
        if roles is None:
            return distribution.log_prob(target)
        return event_log_prob(
            distribution,
            target,
            **_aligned_roles(roles, rows, len(distribution.batch_shape), 1),
        )

    def response_rows(
        self, env: Environment, kept: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The response along its rows, without a run axis of length one.

        Parameters
        ----------
        env : Environment
            The bindings.
        kept : torch.Tensor or None
            The rows kept of a subsampled plate. The response may then be
            given at every row or at the rows kept, in their order.

        Returns
        -------
        torch.Tensor
            The response, ``(rows, *event)``, at the rows kept.

        Raises
        ------
        ValueError
            If the response holds several runs, or neither every row nor
            the rows kept.
        """
        response = env[self.response]
        extents = {self.rows, *(() if kept is None else (int(kept.numel()),))}
        if (
            response.dim() >= 2
            and int(response.shape[0]) not in extents
            and int(response.shape[1]) in extents
        ):
            # The observation carries a leading axis of runs before its rows.
            runs = int(response.shape[0])
            if runs != 1:
                raise ValueError(
                    f"grouped marginalize: observation {self.response!r} has "
                    f"{runs} runs on its leading axis; a grouped block scores "
                    f"one run at a time, so pass one row of input and that "
                    f"run's data"
                )
            response = response[0]
        if kept is None:
            return response
        if int(response.shape[0]) == self.rows:
            return response.index_select(0, kept.to(response.device))
        if int(response.shape[0]) != int(kept.numel()):
            raise ValueError(
                f"observation {self.response!r} has {int(response.shape[0])} rows, "
                f"neither the {self.rows} of plate {self.row_axis!r} nor the "
                f"{int(kept.numel())} kept"
            )
        return response

    def kept_rows(self, conditions: RunConditions) -> torch.Tensor | None:
        """The rows a run under conditions scores.

        Parameters
        ----------
        conditions : RunConditions
            The run's conditions.

        Returns
        -------
        torch.Tensor or None
            The rows kept of the observation's plate, or ``None`` when the
            plate is not subsampled.
        """
        return conditions.subsets.get(self.row_axis)

    def scored_units(self, env: Environment, conditions: RunConditions) -> torch.Tensor:
        """The unit of each row a run under conditions scores.

        Parameters
        ----------
        env : Environment
            The bindings.
        conditions : RunConditions
            The run's conditions.

        Returns
        -------
        torch.Tensor
            ``(rows scored,)``.
        """
        units = self.unit_rows(env)
        kept = self.kept_rows(conditions)
        return units if kept is None else units[kept.to(units.device)]

    def log_prob(
        self,
        env: Environment,
        *,
        powers: bool = True,
        conditions: RunConditions = UNCONDITIONED,
    ) -> torch.Tensor:
        """The log likelihood of each row at each class.

        Parameters
        ----------
        env : Environment
            The bindings.
        powers : bool
            Whether the rows' likelihood powers multiply their scores.
        conditions : RunConditions
            The run's conditions. A fixed response has no factor, so its
            rows score zero; a subsampled plate scores the rows it keeps.

        Returns
        -------
        torch.Tensor
            ``(rows scored, *class_sizes)``.
        """
        kept = self.kept_rows(conditions)
        if self.response in conditions.fixed:
            scored = self.row_count(env) if kept is None else int(kept.numel())
            parts, _ = self._parts(env, self.row_count(env))
            dtype = next(
                (part.dtype for part in parts if part.is_floating_point()),
                torch.get_default_dtype(),
            )
            return torch.zeros(
                scored,
                *self.class_sizes,
                dtype=dtype,
                device=env[self.response].device,
            )
        response = self.response_rows(env, kept)
        rows = int(response.shape[0]) if response.dim() >= 1 else self.rows
        declared_rows = rows if kept is None else self.row_count(env)
        parts, row_parts = self._parts(env, declared_rows)
        scored = rows
        selected = kept
        active: torch.Tensor | None = None
        weights: torch.Tensor | None = None
        if powers and self.weights is not None:
            weights = checked_weights(
                env[self.weights] if isinstance(self.weights, str) else self.weights,
                response,
                declared_rows,
            ).expand(declared_rows)
            if kept is not None:
                weights = weights.index_select(0, kept.to(weights.device))
            active = torch.nonzero(weights > 0, as_tuple=False).flatten()
            if active.numel() == 0:
                zero = (weights * 0.0).sum()
                for part in parts:
                    zero = zero + (part * 0.0).sum()
                return zero.expand(scored, *self.class_sizes)
            if active.numel() == scored:
                # Dense weights need no response or parameter copies.
                active = None
            else:
                response = response.index_select(0, active)
                selected = active if kept is None else kept.to(active.device)[active]
                rows = int(active.numel())
        roles = self._roles(env)
        if selected is not None:
            parts = [
                part.index_select(0, selected.to(part.device)) if is_row else part
                for part, is_row in zip(parts, row_parts, strict=True)
            ]
            roles = (
                None
                if roles is None
                else {
                    name: value.index_select(0, selected.to(value.device))
                    if isinstance(value, torch.Tensor)
                    and value.ndim
                    and int(value.shape[0]) == declared_rows
                    else value
                    for name, value in roles.items()
                }
            )
        if self.crosses:
            distribution = self._builder._dist_builder(
                self._class_parameters(parts, row_parts, rows)
            )
            # Singleton axes after the rows make the response, which does
            # not depend on a class, broadcast against the class axes.
            target = self._typed(response, parts)
            while target.dim() < len(distribution.batch_shape) + len(
                distribution.event_shape
            ):
                target = target.unsqueeze(1)
            score = self._score(distribution, target, roles, rows)
            if score.dim() > 1 + len(self.class_sizes):
                # Each class axis is kept until its own reduction;
                # independent coordinates sum here.
                score = score.reshape(rows, *self.class_sizes, -1).sum(-1)
        else:
            score = self._shared_score(response, parts, roles, rows)
            score = score.reshape(rows, *((1,) * len(self.class_sizes))).expand(
                rows, *self.class_sizes
            )
        if weights is None:
            return score
        shape = (-1, *((1,) * len(self.class_sizes)))
        if active is None:
            return score * weights.reshape(shape)
        score = score * weights.index_select(0, active).reshape(shape)
        return score.new_zeros(scored, *self.class_sizes).index_copy(0, active, score)

    def _typed(self, response: torch.Tensor, parts: list[torch.Tensor]) -> torch.Tensor:
        """The response at the dtype its family's support is scored in.

        Parameters
        ----------
        response : torch.Tensor
            The response.
        parts : list[torch.Tensor]
            The arguments' values.

        Returns
        -------
        torch.Tensor
            The response.
        """
        if not getattr(self.family, "_discrete", False):
            return response
        dtype = next(
            (part.dtype for part in parts if part.is_floating_point()),
            torch.get_default_dtype(),
        )
        return _discrete_value(response, self.family.support, dtype)

    def _shared_score(
        self,
        response: torch.Tensor,
        parts: list[torch.Tensor],
        roles: BoundEvent | None,
        rows: int,
    ) -> torch.Tensor:
        """The log likelihood of each row when no class changes it.

        Parameters
        ----------
        response : torch.Tensor
            The response.
        parts : list[torch.Tensor]
            The arguments' values.
        roles : BoundEvent or None
            The recording roles.
        rows : int
            The number of rows scored.

        Returns
        -------
        torch.Tensor
            ``(rows,)``.
        """
        if self._literal_parameters:
            # Row-only arguments keep their rows even when an intermediate
            # let has no checked plate width: their resolved values reach
            # the builder, which never reads their rows as shared features.
            distribution = self._builder._dist_builder(self._shared_parameters(parts))
            target = self._typed(response, parts)
            while target.dim() < len(distribution.batch_shape) + len(
                distribution.event_shape
            ):
                target = target.unsqueeze(-1)
            score = self._score(distribution, target, roles, rows)
            return score.reshape(rows, -1).sum(-1) if score.dim() > 1 else score
        theta = _stacked(parts)
        if roles is None:
            return self.family.log_prob(theta, response)
        return self.family.log_prob_event(theta, response, roles)

    def _law(
        self, parameters: list[torch.Tensor], roles: BoundEvent | None, rows: int
    ) -> Distribution:
        """The law of the rows at given parameters.

        Parameters
        ----------
        parameters : list[torch.Tensor]
            The family's parameters, each leading with the row axis.
        roles : BoundEvent or None
            The recording roles, whose truncation interval the law is
            normalized on.
        rows : int
            The number of rows.

        Returns
        -------
        Distribution
            A law with one batch axis, the rows; independent coordinates
            of a row are its event.
        """
        law = self._builder._dist_builder(parameters)
        axes = len(law.batch_shape)
        if roles is not None:
            law = truncated_law(law, _aligned_roles(roles, rows, axes, axes))
        return Independent(law, axes - 1) if axes > 1 else law

    def row_count(self, env: Environment) -> int:
        """The number of rows a run draws.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        int
            The length of the index map.
        """
        return int(self.index_tensors(env)[0].shape[0])

    def marginal(
        self, env: Environment, probabilities: Callable[[], torch.Tensor]
    ) -> Distribution:
        """The law of each row with the classes summed out.

        Parameters
        ----------
        env : Environment
            The bindings.
        probabilities : Callable[[], torch.Tensor]
            The probability of each class of each row,
            ``(rows, *class_sizes)``; called only when the law depends on
            the class.

        Returns
        -------
        Distribution
            One distribution per row.
        """
        rows = self.row_count(env)
        parts, row_parts = self._parts(env, rows)
        roles = self._roles(env)
        if not self.crosses:
            if not self._literal_parameters:
                return self.family.sampling_distribution(_stacked(parts), roles)
            return self._law(self._shared_parameters(parts, rows), roles, rows)
        flat = [
            parameter.flatten(1, len(self.class_sizes))
            for parameter in self._class_parameters(parts, row_parts, rows)
        ]
        return Mixture(
            probabilities().flatten(1),
            [
                self._law([parameter[:, index] for parameter in flat], roles, rows)
                for index in range(math.prod(self.class_sizes))
            ],
        )

    def draw(
        self, env: Environment, classes: Callable[[], torch.Tensor]
    ) -> torch.Tensor:
        """Draw the rows at given classes.

        Parameters
        ----------
        env : Environment
            The bindings.
        classes : Callable[[], torch.Tensor]
            The class of each row, counted row-major over the class axes,
            ``(rows,)``; called only when the law depends on the class.

        Returns
        -------
        torch.Tensor
            ``(rows, *event)``.
        """
        rows = self.row_count(env)
        parts, row_parts = self._parts(env, rows)
        roles = self._roles(env)
        if self.crosses:
            chosen = classes()
            positions = torch.arange(rows, device=chosen.device)
            law = self._law(
                [
                    parameter.flatten(1, len(self.class_sizes))[positions, chosen]
                    for parameter in self._class_parameters(parts, row_parts, rows)
                ],
                roles,
                rows,
            )
        elif self._literal_parameters:
            law = self._law(self._shared_parameters(parts, rows), roles, rows)
        else:
            return self.family.rsample_event(_stacked(parts), torch.Size(), roles)
        if self._builder._discrete:
            return _discrete_sample(law, torch.Size(), self.family.support)
        return law.rsample() if law.has_rsample else law.sample()


def _stacked(parts: list[torch.Tensor]) -> torch.Tensor:
    """The input of a family that reads its arguments as one tensor.

    Parameters
    ----------
    parts : list[torch.Tensor]
        The arguments' values.

    Returns
    -------
    torch.Tensor
        The one argument, or the arguments stacked along a last axis.
    """
    if len(parts) == 1:
        return parts[0]
    shape = torch.broadcast_shapes(*(part.shape for part in parts))
    return torch.stack([part.expand(shape) for part in parts], dim=-1)


def _aligned_roles(
    roles: BoundEvent, rows: int, axes: int, position: int
) -> BoundEvent:
    """Lay recording roles given per row out over a law's batch axes.

    Parameters
    ----------
    roles : BoundEvent
        The recording roles.
    rows : int
        The number of rows.
    axes : int
        The number of batch axes of the law.
    position : int
        Where the axes a per-row role lacks are inserted, at most.

    Returns
    -------
    BoundEvent
        The roles.
    """
    laid: BoundEvent = {}
    for name, value in roles.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim
            and int(value.shape[0]) == rows
        ):
            while value.ndim < axes:
                value = value.unsqueeze(min(position, value.ndim))
        laid[name] = value
    return laid


class GroupProjection(ValueModel):
    """How an inner block's units lie in the units of the block around it.

    Parameters
    ----------
    inner_sizes : tuple[int, ...]
        The extent of each factor of the inner plate.
    kept : tuple[int, ...]
        The inner factor that is each factor of the outer plate.
    """

    inner_sizes: tuple[int, ...]
    kept: tuple[int, ...]

    def parents(self, device: torch.device | None = None) -> torch.Tensor:
        """The outer unit each inner unit belongs to.

        Parameters
        ----------
        device : torch.device or None
            Where the result lives.

        Returns
        -------
        torch.Tensor
            ``(inner units,)``.
        """
        positions = torch.arange(math.prod(self.inner_sizes), device=device)
        coordinates = torch.unravel_index(positions, self.inner_sizes)
        return unit_positions(
            tuple(coordinates[axis] for axis in self.kept),
            tuple(self.inner_sizes[axis] for axis in self.kept),
        )


def group_projection(
    inner: tuple[str, ...],
    inner_sizes: tuple[int, ...],
    outer: tuple[str, ...],
    outer_sizes: tuple[int, ...],
) -> GroupProjection | None:
    """The projection of an inner block's plate onto the outer block's.

    Parameters
    ----------
    inner : tuple[str, ...]
        The objects of the inner plate.
    inner_sizes : tuple[int, ...]
        Their extents.
    outer : tuple[str, ...]
        The objects of the outer plate.
    outer_sizes : tuple[int, ...]
        Their extents.

    Returns
    -------
    GroupProjection or None
        The projection when the outer plate's objects are factors of the
        inner plate; the identity of positions when the plates have the
        same extent; ``None`` when the inner plate refines the outer one
        in neither way.
    """
    if all(name in inner for name in outer):
        return GroupProjection(
            inner_sizes=inner_sizes, kept=tuple(inner.index(name) for name in outer)
        )
    extent = math.prod(inner_sizes)
    if extent != math.prod(outer_sizes):
        return None
    return GroupProjection(inner_sizes=(extent,), kept=(0,))


class BlockEntry(ValueModel):
    """One contribution to a grouped block: an observation or an inner block.

    Parameters
    ----------
    slot : str
        The binding that holds the contribution.
    classes : tuple[str, ...]
        The class axes the contribution has after its rows or units.
    observation : GroupedObservation or None
        The observation, for a contribution of rows.
    block : GroupedBlock or None
        The inner block, for a contribution of units.
    projection : GroupProjection or None
        Where the inner block's units lie in this block's.
    """

    slot: str
    classes: tuple[str, ...]
    observation: GroupedObservation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    block: GroupedBlock | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    projection: GroupProjection | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def weighted(self) -> bool:
        """Whether a likelihood power changes the contribution.

        Returns
        -------
        bool
            ``True`` for a weighted observation, or a block holding one.
        """
        if self.observation is not None:
            return self.observation.weights is not None
        assert self.block is not None
        return any(entry.weighted for entry in self.block.entries)

    @property
    def reads(self) -> frozenset[str]:
        """The names the contribution reads as the program binds it.

        Returns
        -------
        frozenset[str]
            The binding that holds it and its index maps.
        """
        maps = () if self.observation is None else self.observation.index_maps
        return frozenset((self.slot, *maps))

    @property
    def raw_reads(self) -> frozenset[str]:
        """The names the contribution reads without likelihood powers.

        Returns
        -------
        frozenset[str]
            The names its observation or its block reads.
        """
        if self.observation is not None:
            return self.observation.score_reads
        assert self.block is not None
        return self.block.raw_reads

    def value(
        self, env: Environment, *, powers: bool, conditions: RunConditions
    ) -> torch.Tensor:
        """The contribution over this block's rows or units.

        Parameters
        ----------
        env : Environment
            The bindings.
        powers : bool
            Whether likelihood powers apply.
        conditions : RunConditions
            The conditions of the run that made the bindings.

        Returns
        -------
        torch.Tensor
            ``(rows, *classes)`` for an observation, and
            ``(units, *classes)`` for an inner block, over this block's
            units.
        """
        if powers or not self.weighted:
            value = env[self.slot]
        elif self.observation is not None:
            value = self.observation.log_prob(env, powers=False, conditions=conditions)
        else:
            assert self.block is not None
            value = self.block.unit_weights(env, powers=False, conditions=conditions)
        if self.projection is None:
            return value
        return _project_nested_aggregates(
            value, self.projection.inner_sizes, self.projection.kept, self.slot
        )


class GroupedBlock:
    """A grouped ``marginalize`` block.

    Parameters
    ----------
    latent : str
        The latent.
    class_count : int
        The number of classes.
    axes : tuple[str, ...]
        The objects of the grouping plate.
    sizes : tuple[int, ...]
        Their extents.
    prior : str
        The binding that holds the class prior.
    prior_indices : tuple[str, ...]
        The latents the prior is gathered at.
    prior_classes : tuple[str, ...]
        The enclosing latents the prior leads with an axis of.
    parameterization : str
        ``"probs"`` or ``"logits"``.
    reduction : Reduction
        How the classes are aggregated.
    parent : GroupedBlock or None
        The block this one lies in.
    projection : GroupProjection or None
        Where this block's units lie in the parent's.
    """

    def __init__(
        self,
        *,
        latent: str,
        class_count: int,
        axes: tuple[str, ...],
        sizes: tuple[int, ...],
        prior: str,
        prior_indices: tuple[str, ...],
        prior_classes: tuple[str, ...],
        parameterization: str,
        reduction: Reduction,
        parent: GroupedBlock | None = None,
        projection: GroupProjection | None = None,
    ) -> None:
        self.latent = latent
        self.class_count = class_count
        self.axes = axes
        self.sizes = sizes
        self.prior = prior
        self.prior_indices = prior_indices
        self.prior_classes = prior_classes
        self.parameterization = parameterization
        self.reduction = reduction
        self.parent = parent
        self.projection = projection
        self.entries: tuple[BlockEntry, ...] = ()
        self._sizes: dict[str, int] = {}

    @property
    def units(self) -> int:
        """The number of units.

        Returns
        -------
        int
            The extent of the grouping plate.
        """
        return math.prod(self.sizes)

    @property
    def chain(self) -> tuple[GroupedBlock, ...]:
        """The blocks around this one and this one, outermost first.

        Returns
        -------
        tuple[GroupedBlock, ...]
            The blocks.
        """
        return (*(self.parent.chain if self.parent is not None else ()), self)

    def add(self, entry: BlockEntry) -> None:
        """Add the next contribution, in the order the program makes it.

        Parameters
        ----------
        entry : BlockEntry
            The contribution.
        """
        self.entries = (*self.entries, entry)

    @property
    def reads(self) -> frozenset[str]:
        """The names the block's score reads.

        Returns
        -------
        frozenset[str]
            The prior's, the contributions', and the index maps'.
        """
        return frozenset({self.prior, *self.prior_indices}).union(
            *(entry.reads for entry in self.entries)
        )

    @property
    def raw_reads(self) -> frozenset[str]:
        """The names the block's score without likelihood powers reads.

        Returns
        -------
        frozenset[str]
            The names the score reads, with those the weighted
            contributions read.
        """
        return self.reads.union(
            *(entry.raw_reads for entry in self.entries if entry.weighted)
        )

    def class_axes(self, class_sizes: Mapping[str, int]) -> tuple[str, ...]:
        """The class axes of the block's contributions and prior.

        Parameters
        ----------
        class_sizes : Mapping[str, int]
            The extent of each latent, in the order the program binds them.

        Returns
        -------
        tuple[str, ...]
            The latents, this block's among them, in that order.
        """
        return tuple(
            name
            for name in class_sizes
            if name == self.latent
            or name in self.prior_classes
            or any(name in entry.classes for entry in self.entries)
        )

    def prior_log_weights(self, env: Environment) -> torch.Tensor:
        """The log prior of the classes at the layout the program binds it.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(*prior classes, *plate, K)``, without the plate when the
            units share one prior.
        """
        weights = env[self.prior]
        for name in self.prior_indices:
            weights = weights[env[name].to(device=weights.device, dtype=torch.long)]
        return _categorical_log_prior(weights, self.parameterization)

    def unit_prior(
        self, env: Environment, class_sizes: Mapping[str, int]
    ) -> torch.Tensor:
        """The log prior of the classes of each unit.

        Parameters
        ----------
        env : Environment
            The bindings.
        class_sizes : Mapping[str, int]
            The class axes of the result and their extents, this block's
            last.

        Returns
        -------
        torch.Tensor
            ``(units, *class extents)``.

        Raises
        ------
        ValueError
            If the prior is laid out over neither the grouping plate nor
            the classes alone.
        """
        prior = self.prior_log_weights(env)
        rank = len(self.prior_classes)
        extents = dict(zip(self.prior_classes, prior.shape[:rank], strict=True))
        extents[self.latent] = self.class_count
        remaining = tuple(int(size) for size in prior.shape[rank:-1])
        if remaining in ((), (1,)):
            prior = prior.reshape(1, *prior.shape[:rank], self.class_count)
        elif remaining in (self.sizes, (self.units,)):
            prior = prior.reshape(
                *prior.shape[:rank], self.units, self.class_count
            ).movedim(rank, 0)
        else:
            raise ValueError(
                "grouped marginalize: conditional prior group axes must be "
                f"{self.sizes}; got {remaining}"
            )
        return prior.reshape(
            prior.shape[0], *(extents.get(name, 1) for name in class_sizes)
        ).expand(self.units, *class_sizes.values())

    def unit_weights(
        self,
        env: Environment,
        *,
        powers: bool = True,
        conditions: RunConditions = UNCONDITIONED,
    ) -> torch.Tensor:
        """The log weight of each unit, its classes aggregated.

        Parameters
        ----------
        env : Environment
            The bindings.
        powers : bool
            Whether the observations' likelihood powers apply.
        conditions : RunConditions
            The conditions of the run that made the bindings.

        Returns
        -------
        torch.Tensor
            ``(units, *enclosing class axes)``.

        Raises
        ------
        ValueError
            If a contribution has no class axis of the block's extent.
        """
        sizes = self._sizes
        values: list[torch.Tensor] = []
        maps: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
        for entry in self.entries:
            value = entry.value(env, powers=powers, conditions=conditions)
            extents = dict(zip(entry.classes, value.shape[1:], strict=True))
            value = value.reshape(
                value.shape[0], *(extents.get(name, 1) for name in sizes)
            ).expand(value.shape[0], *sizes.values())
            if int(value.shape[-1]) != self.class_count:
                raise ValueError(
                    f"grouped marginalize: per-row per-class log-likelihood at "
                    f"slot {entry.slot!r} must end with the class axis of size "
                    f"{self.class_count}; got {tuple(value.shape)}"
                )
            values.append(value)
            if entry.observation is not None:
                kept = entry.observation.kept_rows(conditions)
                maps.append(
                    self._index(
                        tuple(
                            index if kept is None else index[kept.to(index.device)]
                            for index in entry.observation.index_tensors(env)
                        )
                    )
                )
            else:
                # An inner block's aggregates already lie at this block's
                # units.
                maps.append(self._positions(int(value.shape[0]), value.device))
        if self.prior_classes:
            # A conditional prior keeps its enclosing class axes. It is
            # added once per unit, and the reducer's own prior is zero.
            prior = self.unit_prior(env, sizes)
            values.append(prior)
            maps.append(self._positions(self.units, prior.device))
            shared = prior.new_zeros(self.class_count)
        else:
            shared = self.prior_log_weights(env)
        return marginalize_grouped(
            values,
            maps,
            shared,
            self.sizes if len(self.sizes) > 1 else self.sizes[0],
            reduction=self.reduction,
            per_group=True,
        )

    def bind_classes(self, class_sizes: Mapping[str, int]) -> None:
        """Fix the class axes the block aggregates over.

        Parameters
        ----------
        class_sizes : Mapping[str, int]
            The class axes of the block and their extents.
        """
        self._sizes = dict(class_sizes)

    def _index(
        self, maps: tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Index maps in the form the reducer takes them.

        Parameters
        ----------
        maps : tuple[torch.Tensor, ...]
            One index per axis of the plate.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, ...]
            The single index, or the tuple over a product plate.
        """
        return maps if len(self.sizes) > 1 else maps[0]

    def _positions(
        self, count: int, device: torch.device
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """The identity map on the block's units.

        Parameters
        ----------
        count : int
            The number of units.
        device : torch.device
            Where the result lives.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, ...]
            The positions, as one index per axis of a product plate.
        """
        positions = torch.arange(count, dtype=torch.long, device=device)
        if len(self.sizes) == 1:
            return positions
        return self._index(tuple(torch.unravel_index(positions, self.sizes)))

    def row_maps(self) -> Iterator[UnitRows]:
        """The row map of each response of the block and the blocks in it.

        Yields
        ------
        UnitRows
            The map of a response's rows to this block's units.
        """
        for entry in self.entries:
            if entry.observation is not None:
                yield UnitRows(
                    response=entry.observation.response,
                    units=_row_map(entry.observation),
                    one_row_per_unit=(
                        f"If each row has its own {self.latent!r}, state that "
                        f"model in the program: group the block by the rows with "
                        f"`[over={entry.observation.row_axis}]`, and let "
                        "`[via=...]` send each row to itself"
                    ),
                )
                continue
            assert entry.block is not None and entry.projection is not None
            for inner in entry.block.row_maps():
                yield UnitRows(
                    response=inner.response,
                    units=_projected(inner.units, entry.projection),
                    one_row_per_unit=inner.one_row_per_unit,
                )

    def plate_members(self) -> Iterator[PlateMembers]:
        """The plates whose elements lie in the block's units.

        Yields
        ------
        PlateMembers
            The rows of each observation, and the units of each inner
            block along each axis of its plate.
        """
        for entry in self.entries:
            if entry.observation is not None:
                rows = _row_map(entry.observation)
                yield PlateMembers(
                    plate=entry.observation.row_axis,
                    extent=entry.observation.rows,
                    units=rows,
                    positions=_row_positions(rows),
                )
                continue
            assert entry.block is not None and entry.projection is not None
            parents = entry.projection.parents()
            coordinates = torch.unravel_index(
                torch.arange(entry.block.units), entry.block.sizes
            )
            for axis, extent, along in zip(
                entry.block.axes, entry.block.sizes, coordinates, strict=True
            ):
                yield PlateMembers(
                    plate=axis,
                    extent=extent,
                    units=_fixed_map(parents),
                    positions=_fixed_map(along),
                )
            for inner in entry.block.plate_members():
                yield PlateMembers(
                    plate=inner.plate,
                    extent=inner.extent,
                    units=_projected(inner.units, entry.projection),
                    positions=inner.positions,
                )

    def density_obstacle(self) -> str | None:
        """Why a unit's weight is not the log density of its rows.

        Returns
        -------
        str or None
            The reason, or ``None`` when every block sums its classes out.
        """
        if self.reduction != "logsumexp":
            return (
                f"the block over {self.latent!r} aggregates the log weights of "
                f"its classes by {self.reduction}, which is not the log of a "
                "sum over the classes, so its weights are not log densities"
            )
        for entry in self.entries:
            if entry.block is not None:
                reason = entry.block.density_obstacle()
                if reason is not None:
                    return reason
        return None

    def factorization(
        self, conditions: RunConditions = UNCONDITIONED
    ) -> UnitFactorization:
        """The block's score as a sum over its units.

        Parameters
        ----------
        conditions : RunConditions
            The conditions of the run that made the bindings.

        Returns
        -------
        UnitFactorization
            The factorization, whose units are the block's.
        """
        weighted = any(entry.weighted for entry in self.entries)

        def weights(env: Environment) -> torch.Tensor:
            """The log weight of each unit.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            torch.Tensor
                ``(units,)``.
            """
            return self.unit_weights(env, conditions=conditions)

        def raw_weights(env: Environment) -> torch.Tensor:
            """The log weight of each unit without likelihood powers.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            torch.Tensor
                ``(units,)``.
            """
            return self.unit_weights(env, powers=False, conditions=conditions)

        declared = reading(weights, self.reads)
        return _conditioned(
            UnitFactorization(
                latent=self.latent,
                axes=self.axes,
                sizes=self.sizes,
                weights=declared,
                raw_weights=reading(raw_weights, self.raw_reads)
                if weighted
                else declared,
                rows=tuple(self.row_maps()),
                density_obstacle=self.density_obstacle(),
                members=tuple(self.plate_members()),
                under=self.factorization,
            ),
            conditions,
        )

    def score(
        self, env: Environment, conditions: RunConditions = UNCONDITIONED
    ) -> torch.Tensor:
        """The block's contribution to the joint.

        Parameters
        ----------
        env : Environment
            The bindings.
        conditions : RunConditions
            The run's conditions.

        Returns
        -------
        torch.Tensor
            ``(1,)``: the sum of the units' weights, or its estimate from
            the units a subsampled plate keeps.
        """
        if conditions.unconditioned:
            return self.unit_weights(env).sum(dim=0).reshape(1)
        return self.factorization(conditions).weights(env).sum(dim=0).reshape(1)

    def response_law(
        self,
        observation: GroupedObservation,
        class_sizes: Mapping[str, int],
        conditions: RunConditions = UNCONDITIONED,
        earlier: tuple[tuple[BlockEntry, ...], ...] | None = None,
    ) -> ResponseLaw:
        """The law of a response given the responses bound before it.

        Parameters
        ----------
        observation : GroupedObservation
            The response's observation, directly in this block.
        class_sizes : Mapping[str, int]
            The extent of each latent of the program.
        conditions : RunConditions
            The run's conditions. A response an intervention fixes says
            nothing of the classes.
        earlier : tuple[tuple[BlockEntry, ...], ...] or None
            The contributions bound before the response to each block
            around it, outermost first; those made so far when ``None``.

        Returns
        -------
        ResponseLaw
            The law.

        Raises
        ------
        CompileError
            If the observation reads a class of a block that is not around
            it.
        """
        chain = self.chain
        names = tuple(block.latent for block in chain)
        for name in observation.classes:
            if name not in names:
                raise CompileError(
                    f"observation {observation.response!r} reads a value of "
                    f"{name!r}, which a block that does not contain the "
                    "observation sums out",
                    0,
                    0,
                )
        before = tuple(block.entries for block in chain) if earlier is None else earlier
        axes = tuple(
            {name: class_sizes[name] for name in names[: depth + 1]}
            for depth in range(len(chain))
        )
        kept = tuple(names.index(name) for name in observation.classes)

        def marginal(env: Environment) -> Distribution:
            """The law of each row with the classes summed out.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            Distribution
                One distribution per row.
            """
            _require_densities(chain)

            def probabilities() -> torch.Tensor:
                """The class probabilities of each row.

                Returns
                -------
                torch.Tensor
                    ``(rows, *classes of the observation)``.
                """
                joint = chain[0].unit_prior(env, axes[0])
                for block, sizes in zip(chain[1:], axes[1:], strict=True):
                    assert block.projection is not None
                    parents = block.projection.parents(joint.device)
                    joint = joint[parents].unsqueeze(-1) + block.unit_prior(env, sizes)
                summed = tuple(
                    depth + 1 for depth in range(len(chain)) if depth not in kept
                )
                if summed:
                    joint = torch.logsumexp(joint, dim=summed)
                return joint.exp()[observation.unit_rows(env).to(joint.device)]

            return observation.marginal(env, probabilities)

        def draw(env: Environment) -> torch.Tensor:
            """Draw the classes of the units and then the rows.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            torch.Tensor
                ``(rows, *event)``.
            """
            _require_densities(chain)

            def classes() -> torch.Tensor:
                """The class of each row, row-major over its class axes.

                Returns
                -------
                torch.Tensor
                    ``(rows,)``.
                """
                levels = [
                    block._level(env, entries, sizes, conditions)
                    for block, entries, sizes in zip(chain, before, axes, strict=True)
                ]
                drawn = draw_classes(levels)
                rows = observation.unit_rows(env).to(drawn[-1].device)
                return unit_positions(
                    tuple(drawn[depth][rows] for depth in kept),
                    observation.class_sizes,
                )

            return observation.draw(env, classes)

        prior_reads: set[str] = set(observation.parameter_reads)
        for block in chain:
            prior_reads.update((block.prior, *block.prior_indices))
        draw_reads = set(prior_reads)
        for entries in before:
            for entry in entries:
                draw_reads |= entry.reads
                if entry.weighted:
                    draw_reads |= entry.raw_reads

        def under(run: RunConditions) -> ResponseLaw:
            """The law in a run under conditions.

            Parameters
            ----------
            run : RunConditions
                The conditions.

            Returns
            -------
            ResponseLaw
                The law.
            """
            return self.response_law(observation, class_sizes, run, before)

        return ResponseLaw(
            marginal=reading(marginal, prior_reads),
            draw=reading(draw, draw_reads),
            row_axis=observation.row_axis,
            under=under,
        )

    def _level(
        self,
        env: Environment,
        entries: tuple[BlockEntry, ...],
        class_sizes: Mapping[str, int],
        conditions: RunConditions,
    ) -> GroupLevel:
        """The block's class weights given contributions already made.

        Parameters
        ----------
        env : Environment
            The bindings.
        entries : tuple[BlockEntry, ...]
            The contributions.
        class_sizes : Mapping[str, int]
            The class axes of the block and the blocks around it.
        conditions : RunConditions
            The conditions of the run that made the bindings.

        Returns
        -------
        GroupLevel
            The level.
        """
        weights = self.unit_prior(env, class_sizes)
        for entry in entries:
            value = entry.value(env, powers=False, conditions=conditions)
            extents = dict(zip(entry.classes, value.shape[1:], strict=True))
            value = value.reshape(
                value.shape[0], *(extents.get(name, 1) for name in class_sizes)
            ).expand(value.shape[0], *class_sizes.values())
            if entry.observation is None:
                weights = weights + value
            else:
                rows = entry.observation.scored_units(env, conditions)
                weights = weights.index_add(
                    0, rows.to(weights.device), value.to(weights.dtype)
                )
        return GroupLevel(
            log_weights=weights,
            parent=None
            if self.projection is None
            else self.projection.parents(weights.device),
        )


def _require_densities(chain: tuple[GroupedBlock, ...]) -> None:
    """Refuse a law of a block that does not sum its classes out.

    Parameters
    ----------
    chain : tuple[GroupedBlock, ...]
        The blocks around a response.

    Raises
    ------
    ValueError
        If one of them aggregates its classes some other way.
    """
    for block in chain:
        if block.reduction != "logsumexp":
            reason = block.density_obstacle()
            raise ValueError(f"the response has no law to draw from: {reason}")


def _row_map(observation: GroupedObservation) -> Callable[[Environment], torch.Tensor]:
    """The row map of an observation, declaring what it reads.

    Parameters
    ----------
    observation : GroupedObservation
        The observation.

    Returns
    -------
    Callable[[Environment], torch.Tensor]
        The map from the environment to each row's unit.
    """

    def rows(env: Environment) -> torch.Tensor:
        """The unit of each row.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(rows,)``.
        """
        return observation.unit_rows(env)

    return reading(rows, observation.index_maps)


def _row_positions(rows: StepFunction) -> StepFunction:
    """The position of each row along its plate.

    Parameters
    ----------
    rows : StepFunction
        The row map, whose length is the number of rows.

    Returns
    -------
    StepFunction
        From the environment to the rows in order.
    """

    def positions(env: Environment) -> torch.Tensor:
        """The rows in order.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(rows,)``.
        """
        return torch.arange(int(rows(env).shape[0]))

    return _reading_as(positions, rows)


def _fixed_map(values: torch.Tensor) -> StepFunction:
    """A map that reads no binding.

    Parameters
    ----------
    values : torch.Tensor
        Its values.

    Returns
    -------
    StepFunction
        From any environment to the values.
    """

    def fixed(env: Environment) -> torch.Tensor:
        """The values.

        Parameters
        ----------
        env : Environment
            The bindings, which the map does not read.

        Returns
        -------
        torch.Tensor
            The values.
        """
        del env
        return values

    return reading(fixed, ())


def _reading_as(function: StepFunction, other: StepFunction) -> StepFunction:
    """Declare that a function reads the names another one reads.

    Parameters
    ----------
    function : StepFunction
        The function.
    other : StepFunction
        The function whose reads it has.

    Returns
    -------
    StepFunction
        The first function.
    """
    declared = reads_of(other)
    return function if declared is None else reading(function, declared)


def _projected(units: StepFunction, projection: GroupProjection) -> StepFunction:
    """A map into an inner block's units, read in the outer block's.

    Parameters
    ----------
    units : StepFunction
        The map into the inner block's units.
    projection : GroupProjection
        Where the inner units lie in the outer ones.

    Returns
    -------
    StepFunction
        The map from the environment to the outer units.
    """

    def outer(env: Environment) -> torch.Tensor:
        """The outer unit of each element.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(elements,)``.
        """
        inner = units(env)
        return projection.parents(inner.device)[inner]

    return _reading_as(outer, units)


def _conditioned(
    factorization: UnitFactorization, conditions: RunConditions
) -> UnitFactorization:
    """A factorization without the factors and units a run leaves out.

    Parameters
    ----------
    factorization : UnitFactorization
        The factorization, whose weights already leave out the factors of
        fixed responses and the rows a subset does not keep.
    conditions : RunConditions
        The run's conditions.

    Returns
    -------
    UnitFactorization
        The factorization with the row maps of the responses that are not
        fixed and, under a subsampled plate of it, the weight of each unit
        times its coefficient.
    """
    if conditions.fixed:
        factorization = factorization.with_(
            rows=tuple(
                item
                for item in factorization.rows
                if item.response not in conditions.fixed
            )
        )
    if not units_subsampled(factorization, conditions.subsets):
        return factorization
    whole = factorization
    subsets = conditions.subsets

    def weights(env: Environment) -> torch.Tensor:
        """The weight of each unit kept times its coefficient.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(units,)``, zero at a unit not kept, whose own weight is not
            read.
        """
        values = whole.weights(env)
        coefficients = unit_coefficients(whole, env, subsets).to(values)
        kept = (coefficients > 0).nonzero().flatten()
        return torch.zeros_like(values).index_copy(
            0, kept, coefficients[kept] * values[kept]
        )

    return whole.with_(
        weights=weights if whole.reads is None else reading(weights, whole.reads),
        density_obstacle=(
            f"the run scores subsets of the plates {sorted(subsets)}, so its "
            "weights estimate the score and are not the log densities of the units"
        ),
    )


class EnumeratedObservation(ValueModel):
    """The observation of a block scored by running its body at each atom.

    Parameters
    ----------
    response : str
        The observed name.
    family : ContinuousMorphism
        The response family.
    arguments : tuple[StepArgument, ...] or None
        The family's arguments; ``None`` reads the program's input.
    shared_vector : bool
        Whether the family takes one vector-valued parameter, which a
        single shared value presents with a leading axis of one.
    row_axis : str or None
        The object of the observation plate, or ``None`` without a plate.
    placeholder : torch.Tensor or None
        A response of the declared plate's layout; ``None`` for one value
        without a plate.
    array_shape : tuple[int, ...]
        The extents of the independent array each row holds.
    index_map : str or None
        The index map from rows to units; ``None`` when each row is its
        own unit, or every row belongs to the one unit.
    unit_names : frozenset[str]
        The arguments indexed by the units, which the index map reads at
        each row.
    latent_names : frozenset[str]
        The arguments that depend on the latent. They are indexed by the
        units where the latent is drawn, and not at one of its atoms.
    units : int
        The number of units.
    weights : str or float or None
        The likelihood powers of the rows.
    event : ObservationEvent or None
        The recording roles.
    """

    response: str
    family: ContinuousMorphism = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    arguments: tuple[StepArgument, ...] | None = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    shared_vector: bool
    row_axis: str | None = None
    placeholder: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    array_shape: tuple[int, ...] = ()
    index_map: str | None = None
    unit_names: frozenset[str] = frozenset()
    latent_names: frozenset[str] = frozenset()
    units: int = 1
    weights: str | float | None = None
    event: ObservationEvent | None = None

    @property
    def parameter_reads(self) -> frozenset[str]:
        """The names the response law reads.

        Returns
        -------
        frozenset[str]
            The program input, the arguments, the index map, and the
            recording roles.
        """
        names: set[str] = {"_x_input"}
        for argument in self.arguments or ():
            if isinstance(argument, Indexed):
                names.update((argument.name, *argument.indices))
            else:
                names.add(argument)
        if self.index_map is not None:
            names.add(self.index_map)
        if self.event is not None:
            names.update(self.event.names())
        return frozenset(names)

    def declared(self, response: torch.Tensor | None = None) -> ContinuousMorphism:
        """The family laid out along the observation's rows.

        Parameters
        ----------
        response : torch.Tensor or None
            The response whose rows it holds; the declared layout when
            ``None``.

        Returns
        -------
        ContinuousMorphism
            The plated family, or the family itself without a plate.
        """
        if self.placeholder is None:
            return self.family
        return VectorisedObserve(
            self.family,
            self.placeholder if response is None else response,
            array_shape=self.array_shape,
        )

    def kept_rows(self, conditions: RunConditions) -> torch.Tensor | None:
        """The rows a run under conditions scores.

        Parameters
        ----------
        conditions : RunConditions
            The run's conditions.

        Returns
        -------
        torch.Tensor or None
            The rows kept of the observation's plate, or ``None`` when it
            has none or the plate is not subsampled.
        """
        if self.row_axis is None:
            return None
        return conditions.subsets.get(self.row_axis)

    def value(self, env: Environment, kept: torch.Tensor | None = None) -> torch.Tensor:
        """The response without a run axis of length one before its rows.

        Parameters
        ----------
        env : Environment
            The bindings.
        kept : torch.Tensor or None
            The rows kept of a subsampled plate, at which the response may
            be given.

        Returns
        -------
        torch.Tensor
            The response, at the layout it is given in.
        """
        response = env[self.response]
        if self.placeholder is None:
            return response
        extents = {
            int(self.placeholder.shape[0]),
            *(() if kept is None else (int(kept.numel()),)),
        }
        if (
            response.dim() >= 2
            and int(response.shape[0]) == 1
            and 1 not in extents
            and int(response.shape[1]) in extents
        ):
            return response[0]
        return response

    def row_count(self) -> int:
        """The number of rows of the observation.

        Returns
        -------
        int
            The extent of the response's plate; one without a plate.
        """
        return 1 if self.placeholder is None else int(self.placeholder.shape[0])

    def parameter(self, env: Environment, *, drawn: bool = False) -> torch.Tensor:
        """The family's parameter at one value of the latent.

        Parameters
        ----------
        env : Environment
            The bindings, the latent's and the body's among them.
        drawn : bool
            Whether the latent holds one drawn value per unit, rather than
            one atom for all of them.

        Returns
        -------
        torch.Tensor
            One argument as it is, or several stacked along a last axis.
        """
        local = env
        if self.index_map is not None:
            rows = _index_map_tensor(env, self.index_map).reshape(-1).to(torch.long)
            local = dict(env)
            names = self.unit_names | (self.latent_names if drawn else frozenset())
            for name in names:
                value = env.get(name)
                if (
                    value is not None
                    and value.dim()
                    and int(value.shape[0]) == self.units
                ):
                    local[name] = value[rows.to(value.device)]
        return _observation_parameter(local, self.arguments, self.shared_vector)

    def roles(self, env: Environment) -> BoundEvent | None:
        """The recording roles at one run's values.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        BoundEvent or None
            The roles, or ``None`` for an observation recorded exactly.
        """
        return None if self.event is None else resolve_event(self.event, env)

    def row_scores(
        self,
        env: Environment,
        *,
        powers: bool,
        conditions: RunConditions = UNCONDITIONED,
    ) -> torch.Tensor:
        """The log likelihood of each row at one atom of the latent.

        Parameters
        ----------
        env : Environment
            The bindings, the latent's and the body's among them.
        powers : bool
            Whether the rows' likelihood powers multiply their scores.
        conditions : RunConditions
            The run's conditions. A subsampled plate scores the rows it
            keeps.

        Returns
        -------
        torch.Tensor
            ``(rows scored,)``, with one entry for an observation without
            a plate.
        """
        kept = self.kept_rows(conditions)
        response = self.value(env, kept)
        theta = self.parameter(env)
        weights = self.weights if powers else None
        if kept is not None:
            return self._kept_scores(env, theta, response, weights, kept)
        if weights is None and self.event is None:
            score = self.family.log_prob(theta, response)
        else:
            declared = self.declared(response)
            family: ContinuousMorphism = declared
            roles = self.roles(env)
            if roles is not None:
                family = RecordedObservation(family, declared, roles)
            if weights is not None:
                family = WeightedObservation(
                    family,
                    declared,
                    env[weights] if isinstance(weights, str) else weights,
                    theta,
                )
            pointwise = getattr(family, "log_prob_pointwise", None)
            score = (
                pointwise(theta, response)
                if callable(pointwise)
                else family.log_prob(theta, response)
            )
        assert isinstance(score, torch.Tensor)
        if self.placeholder is None:
            return score.sum().reshape(1)
        return score if score.dim() == 1 else score.reshape(score.shape[0], -1).sum(-1)

    def _kept_scores(
        self,
        env: Environment,
        theta: torch.Tensor,
        response: torch.Tensor,
        weights: str | float | None,
        kept: torch.Tensor,
    ) -> torch.Tensor:
        """The log likelihood of the rows a subsampled plate keeps.

        Parameters
        ----------
        env : Environment
            The bindings at one atom of the latent.
        theta : torch.Tensor
            The family's parameter, shared or one row per observation row.
        response : torch.Tensor
            The response, at every row or at the rows kept.
        weights : str or float or None
            The likelihood powers that apply.
        kept : torch.Tensor
            The rows kept.

        Returns
        -------
        torch.Tensor
            ``(rows kept,)``.
        """
        declared = self.declared()
        assert isinstance(declared, VectorisedObserve)
        count = int(kept.numel())
        if int(response.shape[0]) != count:
            response = response.index_select(0, kept.to(response.device))
        powers: torch.Tensor | None = None
        active = torch.arange(count)
        if weights is not None:
            powers = checked_weights(
                env[weights] if isinstance(weights, str) else weights,
                theta,
                int(declared.response.shape[0]),
            ).expand(int(declared.response.shape[0]))
            powers = powers.index_select(0, kept.to(powers.device))
            # A row with power zero has no factor, whatever its density.
            active = (powers > 0).nonzero().flatten().cpu()
            if not active.numel():
                return (theta.sum() * 0.0 + powers.sum() * 0.0).expand(count)
        scored = kept.cpu()[active]
        response = response.index_select(0, active.to(response.device))
        arrays = tuple(torch.arange(extent) for extent in self.array_shape)
        roles = self.roles(env)
        if theta.dim() and int(theta.shape[0]) not in (
            1,
            int(declared.response.shape[0]),
        ):
            # A shared parameter without a row axis.
            theta = theta.unsqueeze(0)
        score = (
            declared.log_prob_pointwise_rows(theta, response, scored, *arrays)
            if roles is None
            else declared.log_prob_pointwise_rows_event(
                theta, response, roles, scored, *arrays
            )
        )
        assert isinstance(score, torch.Tensor)
        score = score.reshape(int(active.numel()), -1).sum(-1)
        if powers is None:
            return score
        active = active.to(score.device)
        return score.new_zeros(count).index_copy(
            0, active, score * powers[active.to(powers.device)]
        )

    def marginal(
        self, envs: list[Environment], probabilities: torch.Tensor
    ) -> Distribution:
        """The law of each row with the latent summed out.

        Parameters
        ----------
        envs : list[Environment]
            The bindings at each atom of the latent.
        probabilities : torch.Tensor
            The probability of each atom, ``(atoms,)`` or
            ``(rows, atoms)``.

        Returns
        -------
        Distribution
            One distribution per row and entry of its array.
        """
        declared = self.declared()
        entries = math.prod(self.array_shape)
        if probabilities.dim() > 1 and entries > 1:
            probabilities = probabilities.repeat_interleave(entries, dim=0)
        laws: list[Distribution] = []
        for env in envs:
            law = declared.sampling_distribution(self.parameter(env), self.roles(env))
            rows = 0 if self.placeholder is None else int(self.placeholder.shape[0])
            if len(law.batch_shape) > 1 and int(law.batch_shape[0]) == rows:
                law = Independent(law, len(law.batch_shape) - 1)
            laws.append(law)
        return Mixture(probabilities, laws)

    def draw(self, env: Environment) -> torch.Tensor:
        """Draw the rows at the atom the run drew.

        Parameters
        ----------
        env : Environment
            The bindings, the drawn latent's and the body's among them.

        Returns
        -------
        torch.Tensor
            The response at its declared layout.
        """
        declared = self.declared()
        theta = self.parameter(env, drawn=True)
        roles = self.roles(env)
        drawn = (
            declared.rsample(theta)
            if roles is None
            else declared.rsample_event(theta, torch.Size(), roles)
        )
        if self.placeholder is not None and drawn.numel() == self.placeholder.numel():
            return drawn.reshape(self.placeholder.shape)
        return drawn


class EnumeratedBlock(ValueModel):
    """A block whose body runs once at each atom of its latent.

    With no unit axis the block has one latent, which all the rows of its
    observation share. A block indexed by a plate has one latent per
    element of it.

    Parameters
    ----------
    latent : str
        The latent.
    axes : tuple[str, ...]
        The object that indexes the latent, or nothing for one latent.
    sizes : tuple[int, ...]
        Its extent.
    prior : StepArgument
        The prior's parameter.
    binary : bool
        Whether the atoms are zero and one, rather than the classes of the
        prior.
    parameterization : str or None
        ``"probs"`` or ``"logits"``.
    reduction : Reduction
        How the atoms are aggregated.
    lets : tuple[tuple[str, Callable[[Environment], torch.Tensor]], ...]
        The body's deterministic bindings, in order.
    observation : EnumeratedObservation or None
        The body's observation.
    body_reads : frozenset[str]
        The names the body reads.
    """

    latent: str
    axes: tuple[str, ...]
    sizes: tuple[int, ...]
    prior: StepArgument = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    binary: bool
    parameterization: str | None
    reduction: Reduction
    lets: tuple[tuple[str, Callable[[Environment], torch.Tensor]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observation: EnumeratedObservation | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    body_reads: frozenset[str] = frozenset()

    @property
    def units(self) -> int:
        """The number of units, one per latent.

        Returns
        -------
        int
            The extent of the indexing object, or one.
        """
        return math.prod(self.sizes)

    def prior_value(self, env: Environment) -> torch.Tensor:
        """The prior's parameter, one value per unit.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            The parameter at its own layout; a binary prior that every
            unit shares is laid out along the units.
        """
        value = _lookup_arg(env, self.prior)
        if self.binary and self.axes and value.numel() == 1:
            return value.reshape(1).expand(self.units)
        return value

    def log_prior(self, env: Environment) -> torch.Tensor:
        """The log prior of each atom.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(units, atoms)``.

        Raises
        ------
        ValueError
            If the prior does not hold one distribution, or one per unit.
        """
        value = self.prior_value(env)
        weights = (
            _binary_log_prior(value, self.parameterization)
            if self.binary
            else _categorical_log_prior(value, self.parameterization)
        )
        flat = weights.reshape(-1, weights.shape[-1])
        if int(flat.shape[0]) not in (1, self.units):
            holder = (
                f"one prior per element of {self.axes[0]!r}"
                if self.axes
                else "one prior, since it draws one value"
            )
            raise ValueError(
                f"marginalize {self.latent!r}: the block takes {holder}; its "
                f"prior holds {int(flat.shape[0])}, with shape {tuple(value.shape)}"
            )
        return flat.expand(self.units, flat.shape[-1])

    def atoms(self, env: Environment) -> list[Environment]:
        """The bindings of the body at each atom of the latent.

        Parameters
        ----------
        env : Environment
            The bindings before the block.

        Returns
        -------
        list[Environment]
            One environment per atom.
        """
        value = _lookup_arg(env, self.prior)
        if self.binary:
            values = [
                torch.zeros((), dtype=value.dtype, device=value.device),
                torch.ones((), dtype=value.dtype, device=value.device),
            ]
        else:
            values = [
                torch.tensor(index, dtype=torch.long, device=value.device)
                for index in range(int(value.shape[-1]))
            ]
        environments: list[Environment] = []
        for atom in values:
            local = dict(env)
            local[self.latent] = atom
            for name, function in self.lets:
                local[name] = function(local)
            environments.append(local)
        return environments

    def unit_rows(self, env: Environment) -> torch.Tensor:
        """The unit each row of the response belongs to.

        Parameters
        ----------
        env : Environment
            The bindings.

        Returns
        -------
        torch.Tensor
            ``(rows,)``.
        """
        assert self.observation is not None
        if self.axes and self.observation.index_map is not None:
            index = _index_map_tensor(env, self.observation.index_map)
            return unit_positions((index.reshape(-1).to(torch.long),), self.sizes)
        device = env[self.observation.response].device
        rows = self.observation.row_count()
        if self.axes:
            return torch.arange(rows, device=device)
        return torch.zeros(rows, dtype=torch.long, device=device)

    def unit_weights(
        self,
        env: Environment,
        *,
        powers: bool = True,
        conditions: RunConditions = UNCONDITIONED,
    ) -> torch.Tensor:
        """The log weight of each unit, its atoms aggregated.

        Parameters
        ----------
        env : Environment
            The bindings before the block, with the response.
        powers : bool
            Whether the observation's likelihood powers apply.
        conditions : RunConditions
            The run's conditions. A fixed response has no factor; a
            subsampled plate scores the rows it keeps.

        Returns
        -------
        torch.Tensor
            ``(units,)``.

        Raises
        ------
        ValueError
            If the block has one latent per element of an object, and its
            observation does not hold one row per element.
        """
        weighted = self.log_prior(env)
        observation = self.observation
        if observation is not None and observation.response not in conditions.fixed:
            scores = torch.stack(
                [
                    observation.row_scores(atom, powers=powers, conditions=conditions)
                    for atom in self.atoms(env)
                ],
                dim=-1,
            )
            kept = observation.kept_rows(conditions)
            if not self.axes:
                weighted = weighted + scores.sum(dim=0, keepdim=True)
            elif kept is not None:
                rows = self.unit_rows(env)
                weighted = weighted.index_add(
                    0, rows[kept.to(rows.device)], scores.to(weighted.dtype)
                )
            elif observation.index_map is None:
                if int(scores.shape[0]) != self.units:
                    raise ValueError(
                        f"marginalize {self.latent!r}: the block draws one value "
                        f"per element of {self.axes[0]!r}, of which there are "
                        f"{self.units}, and its observation "
                        f"{observation.response!r} holds {int(scores.shape[0])} "
                        "rows; give the observation an index map with "
                        "`[via=...]`"
                    )
                weighted = weighted + scores
            else:
                weighted = weighted.index_add(
                    0, self.unit_rows(env), scores.to(weighted.dtype)
                )
        if self.reduction == "logsumexp":
            return torch.logsumexp(weighted, dim=-1)
        if self.reduction == "sum":
            return weighted.sum(dim=-1)
        return weighted.mean(dim=-1)

    def density_obstacle(self) -> str | None:
        """Why a unit's weight is not the log density of its rows.

        Returns
        -------
        str or None
            The reason, or ``None`` when the atoms are summed out.
        """
        if self.reduction == "logsumexp":
            return None
        return (
            f"the block over {self.latent!r} aggregates the log weights of its "
            f"classes by {self.reduction}, which is not the log of a sum over "
            "the classes, so its weights are not log densities"
        )

    @property
    def score_reads(self) -> frozenset[str]:
        """The names the score reads.

        Returns
        -------
        frozenset[str]
            The prior's and the body's names, the response, and the
            likelihood powers.
        """
        names: set[str] = {"_x_input", *self.body_reads, *self._prior_reads}
        if self.observation is not None:
            names |= {self.observation.response, *self.observation.parameter_reads}
            if isinstance(self.observation.weights, str):
                names.add(self.observation.weights)
        return frozenset(names)

    @property
    def _prior_reads(self) -> frozenset[str]:
        """The names the prior reads.

        Returns
        -------
        frozenset[str]
            The parameter's binding and the indices it is gathered at.
        """
        if isinstance(self.prior, Indexed):
            return frozenset((self.prior.name, *self.prior.indices))
        return frozenset((self.prior,))

    def score(
        self, env: Environment, conditions: RunConditions = UNCONDITIONED
    ) -> torch.Tensor:
        """The block's contribution to the joint.

        Parameters
        ----------
        env : Environment
            The bindings.
        conditions : RunConditions
            The run's conditions.

        Returns
        -------
        torch.Tensor
            ``(1,)``: the sum of the units' weights, or its estimate from
            the units a subsampled plate keeps.
        """
        if conditions.unconditioned:
            return self.unit_weights(env).sum().reshape(1)
        return self.factorization(conditions).weights(env).sum().reshape(1)

    def factorization(
        self, conditions: RunConditions = UNCONDITIONED
    ) -> UnitFactorization:
        """The block's score as a sum over its units.

        Parameters
        ----------
        conditions : RunConditions
            The run's conditions.

        Returns
        -------
        UnitFactorization
            One unit per element of the indexing object, or one unit for a
            block with one latent.
        """

        def weights(env: Environment) -> torch.Tensor:
            """The log weight of each unit.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            torch.Tensor
                ``(units,)``.
            """
            return self.unit_weights(env, conditions=conditions)

        def raw_weights(env: Environment) -> torch.Tensor:
            """The log weight of each unit without likelihood powers.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            torch.Tensor
                ``(units,)``.
            """
            return self.unit_weights(env, powers=False, conditions=conditions)

        rows: tuple[UnitRows, ...] = ()
        members: tuple[PlateMembers, ...] = ()
        weighted = False
        if self.observation is not None:
            observation = self.observation
            weighted = observation.weights is not None
            names = {observation.response}
            if observation.index_map is not None:
                names.add(observation.index_map)

            def row_map(env: Environment) -> torch.Tensor:
                """The unit of each row.

                Parameters
                ----------
                env : Environment
                    The bindings.

                Returns
                -------
                torch.Tensor
                    ``(rows,)``.
                """
                return self.unit_rows(env)

            units = reading(row_map, names)
            rows = (
                UnitRows(
                    response=observation.response,
                    units=units,
                    one_row_per_unit=self._one_row_per_unit(observation.row_axis),
                ),
            )
            if observation.row_axis is not None:
                members = (
                    PlateMembers(
                        plate=observation.row_axis,
                        extent=observation.row_count(),
                        units=units,
                        positions=_row_positions(units),
                    ),
                )
        declared = reading(weights, self.score_reads)
        return _conditioned(
            UnitFactorization(
                latent=self.latent,
                axes=self.axes,
                sizes=self.sizes,
                weights=declared,
                raw_weights=reading(raw_weights, self.score_reads)
                if weighted
                else declared,
                rows=rows,
                density_obstacle=self.density_obstacle(),
                members=members,
                under=self.factorization,
            ),
            conditions,
        )

    def _one_row_per_unit(self, row_axis: str | None) -> str:
        """The program that gives each row of the observation its own latent.

        Parameters
        ----------
        row_axis : str or None
            The object of the observation plate.

        Returns
        -------
        str
            The change, as a sentence.
        """
        plate = "the rows" if row_axis is None else row_axis
        if self.binary:
            form = (
                f"index the latent by the rows, `marginalize {self.latent} : {plate}`"
            )
        else:
            form = (
                f"group the block by the rows with `[over={plate}]`, and give the "
                "observation an index map, `[via=...]`, that sends each row to "
                "itself"
            )
        return (
            f"If each row has its own {self.latent!r}, state that model in the "
            f"program: {form}"
        )

    def response_law(self) -> ResponseLaw:
        """The law of the block's response.

        A draw reads the latent the run drew before the block, so the
        response agrees with the latent and the bindings of the same run.

        Returns
        -------
        ResponseLaw
            The law.
        """
        observation = self.observation
        assert observation is not None

        def marginal(env: Environment) -> Distribution:
            """The law of each row with the latent summed out.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            Distribution
                One distribution per row.

            Raises
            ------
            ValueError
                If the block does not sum its latent out.
            """
            reason = self.density_obstacle()
            if reason is not None:
                raise ValueError(f"the response has no marginal law: {reason}")
            probabilities = self.log_prior(env).exp()
            if not self.axes:
                probabilities = probabilities[0]
            elif observation.index_map is not None:
                probabilities = probabilities[self.unit_rows(env)]
            return observation.marginal(self.atoms(env), probabilities)

        def draw(env: Environment) -> torch.Tensor:
            """Draw the rows at the latent the run drew.

            Parameters
            ----------
            env : Environment
                The bindings.

            Returns
            -------
            torch.Tensor
                ``(rows, *event)``.
            """
            return observation.draw(env)

        marginal_reads = {
            *self._prior_reads,
            *self.body_reads,
            *observation.parameter_reads,
        }
        return ResponseLaw(
            marginal=reading(marginal, marginal_reads),
            draw=reading(
                draw,
                {
                    *observation.parameter_reads,
                    self.latent,
                },
            ),
            row_axis=observation.row_axis,
        )


__all__ = [
    "BlockEntry",
    "EnumeratedBlock",
    "EnumeratedObservation",
    "GroupProjection",
    "GroupedBlock",
    "GroupedObservation",
    "group_projection",
    "unit_positions",
]
