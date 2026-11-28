"""Typed fixed-word posterior laws over sparse original path traces."""

from __future__ import annotations


import didactic.api as dx
import torch

from quivers._positions import depends_on_position
from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    FixedStringQuery,
    ProductiveEpsilonCycleError,
    RelationTopology,
    Words,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import sampleable_type, tensor_type
from quivers.qiec.kinds import NAT
from quivers.qiec.numbers import PositionDependentControl, host_index
from quivers.qiec.relations import (
    RationalRelationSpec,
    _validate_fixed_observation,
    RelationWeightBindings,
)
from quivers.qiec.types import INT, IndexLiteral, TypeExpr, product_type


def _check_trace_capacity(topology: RelationTopology, capacity: int) -> None:
    """Check the static trace carrier against the original boundary states.

    Parameters
    ----------
    topology : RelationTopology
        Original path states and arcs.
    capacity : int
        Number of positions in the arc occurrence tensor.

    Raises
    ------
    ValueError
        The tensor extent or an original boundary state cannot be represented.
    """
    if type(capacity) is not int or capacity < 0:
        raise ValueError("trace_capacity must be a nonnegative integer")
    maximum = torch.iinfo(torch.int64).max
    if any(mark.state > maximum for mark in (*topology.initial, *topology.final)):
        raise ValueError("path boundary states must fit in the QIEC Int carrier")


def _trace_element_type(capacity: int) -> TypeExpr:
    """Type the ordered state fields and occurrence vector of a trace.

    Parameters
    ----------
    capacity : int
        Number of positions in the arc occurrence tensor.

    Returns
    -------
    TypeExpr
        The canonical product type.
    """
    extent = IndexLiteral(value=capacity, sort=NAT)
    return product_type(INT, INT, tensor_type(INT, (extent,)))


class RelationPathLaw:
    """Draw and score fixed-word posterior paths in a checked trace type.

    Parameters
    ----------
    query : FixedStringQuery
        The observed path graph with live owned weights.
    trace_capacity : int
        Number of arc positions in a padded trace. Acyclic support must
        fit. A permitted cyclic draw fails if it needs another arc.
    generator : torch.Generator
        The CPU generator supplying each Random draw.
    allow_cyclic : bool
        Permit a certified cyclic fixed-word query with a per-draw
        resource ceiling and full posterior trace scores.
    """

    def __init__(
        self,
        query: FixedStringQuery,
        *,
        trace_capacity: int,
        generator: torch.Generator,
        allow_cyclic: bool = False,
    ) -> None:
        _check_trace_capacity(query._relation.topology, trace_capacity)
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("a sparse path law needs a CPU generator")
        self._cyclic = query._walks._cyclic
        longest = None if allow_cyclic and self._cyclic else query.max_path_length
        if longest is not None and longest > trace_capacity:
            raise ValueError(
                "trace_capacity is smaller than a structurally accepting path"
            )
        with torch.no_grad():
            query._path_log_normalizer(query._log_path_weights())
        self._query = query
        self._capacity = trace_capacity
        self._generator = generator
        self._element = _trace_element_type(trace_capacity)

    @property
    def has_productive_cycle(self) -> bool:
        """Whether the observation has structurally unbounded path support.

        Returns
        -------
        bool
            True when a productive cycle can admit unbounded walks as
            the original live weights change.
        """
        return self._cyclic

    @property
    def element(self) -> TypeExpr:
        """Return the exact checked type of one complete path.

        Returns
        -------
        TypeExpr
            A product with two states and the padded arc vector.
        """
        return self._element

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw a trace using the law's current weights and CPU generator.

        Returns
        -------
        tuple of torch.Tensor
            The initial state, final state, and padded original arc IDs.

        Raises
        ------
        RelationLimitError
            A cyclic sample attempted to traverse past the trace capacity.
        """
        with torch.no_grad():
            path = self._query._walks.sample(
                self._query._log_path_weights(dtype=torch.float64),
                LOG_PROB,
                self._generator,
                self._capacity,
            )
        padding = (-1,) * (self._capacity - len(path.arc_ids))
        values = path.initial_state, path.final_state, (*path.arc_ids, *padding)
        return (
            torch.tensor(values[0], dtype=torch.int64, device=path.weight.device),
            torch.tensor(values[1], dtype=torch.int64, device=path.weight.device),
            torch.tensor(values[2], dtype=torch.int64, device=path.weight.device),
        )

    def log_prob(self, value: object, keep_batch: bool = False) -> torch.Tensor:
        """Score one trace in the fixed-word posterior.

        Parameters
        ----------
        value : object
            The initial state, final state, and padded original arc IDs
            in the checked machine's representation.
        keep_batch : bool
            This unbatched law accepts False.

        Returns
        -------
        torch.Tensor
            The differentiable log probability, or negative infinity
            for a well-typed value outside the support.

        Raises
        ------
        ValueError
            A batch or a value outside the machine carrier was supplied.
        """
        if keep_batch:
            raise ValueError("a sparse path law has no batch axes")
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError(
                "a sparse path trace must have two states and an arc tensor"
            )
        initial = host_index(value[0], "path initial state")
        final = host_index(value[1], "path final state")
        arcs = value[2]
        if (
            not isinstance(arcs, torch.Tensor)
            or arcs.dtype != torch.int64
            or arcs.shape != (self._capacity,)
        ):
            raise ValueError(
                "path arc occurrence tensor does not match its Int capacity"
            )
        if depends_on_position(arcs):
            raise PositionDependentControl("path arc occurrence tensor")
        ids = tuple(arcs.tolist())
        if -1 in ids:
            stop = ids.index(-1)
            if all(arc_id == -1 for arc_id in ids[stop:]):
                ids = ids[:stop]
        return self._query.path_log_prob(initial, ids, final)

    def log_mass(self, batch_axes: int = 0) -> torch.Tensor:
        """Return zero when the trace carrier represents the whole law.

        Parameters
        ----------
        batch_axes : int
            The retained batch rank, which must be zero.

        Returns
        -------
        torch.Tensor
            Scalar log total mass in the underlying number domain.

        Raises
        ------
        ValueError
            A batch axis was requested or the current normalizer is invalid.
        ProductiveEpsilonCycleError
            The bounded trace carrier does not represent the whole cyclic law.
        """
        if type(batch_axes) is not int or batch_axes != 0:
            raise ValueError("a sparse path law has no batch axes")
        if self._cyclic:
            raise ProductiveEpsilonCycleError(
                "log_mass of a cyclic path law is unavailable on its bounded trace carrier"
            )
        weights = self._query._log_path_weights()
        self._query._path_log_normalizer(weights)
        return weights.new_zeros(())


class FixedRelationPathSampleRequest(QiecModel):
    """A sparse posterior law whose typed values identify complete paths.

    Paths are recorded as the pair of original boundary states and a
    vector of original arc occurrences. Its capacity is finite.
    Trailing positions use -1 padding. The reachable and coaccessible
    fixed-word graph must be acyclic, and its longest accepting path
    must fit the capacity, even if the path currently has zero weight.

    Parameters
    ----------
    relation : RationalRelationSpec
        The original typed Markov or LogProb topology and weight identities.
    words : Words
        One observed word per tape in tape order.
    trace_capacity : int
        The fixed number of positions in the result's arc tensor.
    max_states : int
        The state budget for constructing the fixed-word graph.
    max_work : int
        The arc examination budget for constructing the fixed-word graph.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    trace_capacity: int
    max_states: int
    max_work: int

    @dx.model_validator()
    def _validate(self) -> None:
        """Check the observation, budgets, and representable trace fields.

        Raises
        ------
        ValueError
            Invalid words, limits, capacity, or integer boundary range.
        """
        _validate_fixed_observation(
            self.relation, self.words, self.max_states, self.max_work, None
        )
        _check_trace_capacity(self.relation.topology, self.trace_capacity)

    @property
    def element_type(self) -> TypeExpr:
        """Return the checked type of one complete path trace.

        Returns
        -------
        TypeExpr
            A product of two Int states and a tensor of Int arc IDs.
        """
        return _trace_element_type(self.trace_capacity)

    @property
    def result_type(self) -> TypeExpr:
        """Return the type of the posterior law.

        Returns
        -------
        TypeExpr
            The Sampleable instance over the declared trace carrier.
        """
        return sampleable_type(self.element_type)

    def bind(
        self,
        parameters: RelationWeightBindings,
        *,
        generator: torch.Generator,
    ) -> RelationPathLaw:
        """Bind the law for a checked Random.sample operation.

        Parameters
        ----------
        parameters : RelationWeightBindings
            Original live scalar parameter owners.
        generator : torch.Generator
            The CPU generator used for posterior path draws.

        Returns
        -------
        RelationPathLaw
            A reusable Sampleable of the declared trace type.

        Raises
        ------
        ValueError
            The generator is unsupported, the trace capacity is too
            small, or the observed probability mass is invalid.
        """
        query = self.relation.bind(parameters).query(
            self.words, max_states=self.max_states, max_work=self.max_work
        )
        return RelationPathLaw(
            query, trace_capacity=self.trace_capacity, generator=generator
        )


class FixedCyclicRelationPathSampleRequest(QiecModel):
    """Admit certified cyclic sampling with bounded complete trace encoding.

    The existing trace capacity also bounds the number of drawn arcs.
    A draw that needs another arc raises a resource error. Each
    returned trace records a complete original walk and receives its
    full posterior log probability. A bound on the execution does not
    truncate or renormalize the model distribution.

    Parameters
    ----------
    request : FixedRelationPathSampleRequest
        The original typed observation, trace capacity and construction limits.
    max_closure_size : int
        Maximum local matrix order for a productive certified closure.
    """

    request: FixedRelationPathSampleRequest = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_closure_size: int

    @dx.model_validator()
    def _validate_closure_limit(self) -> None:
        """Check the explicit structural closure limit.

        Raises
        ------
        ValueError
            The closure capacity is invalid.
        """
        _validate_limits(None, None, self.max_closure_size)

    @property
    def element_type(self) -> TypeExpr:
        """Return the original complete trace type.

        Returns
        -------
        TypeExpr
            A product of two Int states and a tensor of original arc IDs.
        """
        return self.request.element_type

    @property
    def result_type(self) -> TypeExpr:
        """Return the checked Sampleable type for complete traces.

        Returns
        -------
        TypeExpr
            The Sampleable instance over the original trace carrier.
        """
        return self.request.result_type

    def bind(
        self, parameters: RelationWeightBindings, *, generator: torch.Generator
    ) -> RelationPathLaw:
        """Bind the original live weights to a closure-checked Random law.

        Parameters
        ----------
        parameters : RelationWeightBindings
            The original live weight owners.
        generator : torch.Generator
            The CPU generator for each attempted draw.

        Returns
        -------
        RelationPathLaw
            The bounded draw operation with full posterior trace scores.

        Raises
        ------
        ValueError
            The generator, binding, capacity, or normalizer is invalid.
        RelationLimitError
            Observation construction or closure exceeds its limit.
        """
        source = self.request
        query = source.relation.bind(parameters).query(
            source.words,
            max_states=source.max_states,
            max_work=source.max_work,
            max_closure_size=self.max_closure_size,
        )
        return RelationPathLaw(
            query,
            trace_capacity=source.trace_capacity,
            generator=generator,
            allow_cyclic=True,
        )


__all__ = [
    "FixedCyclicRelationPathSampleRequest",
    "FixedRelationPathSampleRequest",
    "RelationPathLaw",
]
