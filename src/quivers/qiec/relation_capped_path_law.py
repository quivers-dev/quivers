"""A checked pushforward onto bounded traces and one cyclic path tail."""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers._positions import depends_on_position
from quivers._values import OPAQUE_BY_VALUE
from quivers.core.relations._capped_paths import _CappedPaths
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import sampleable_type, tensor_type
from quivers.qiec.kinds import NAT
from quivers.qiec.numbers import PositionDependentControl, host_index
from quivers.qiec.relation_path_law import FixedCyclicRelationPathSampleRequest
from quivers.qiec.relations import RelationWeightBindings
from quivers.qiec.types import BOOL, INT, IndexLiteral, TypeExpr, product_type


def _capped_trace_type(capacity: int) -> TypeExpr:
    """Type a tagged original trace or a canonical tail record.

    Parameters
    ----------
    capacity : int
        The fixed number of arc positions.

    Returns
    -------
    TypeExpr
        The tagged product of the Bool flag, states, and arc tensor.
    """
    extent = IndexLiteral(value=capacity, sort=NAT)
    return product_type(BOOL, INT, INT, tensor_type(INT, (extent,)))


class CappedRelationPathLaw:
    """Push a fixed-word posterior onto retained original traces and one tail.

    A true tag denotes the tail and accompanies -1 in every trace field.
    A false tag denotes a complete original trace, with -1 suffix padding.
    The tail contains every accepting path with more than the capacity's
    original arc traversals. The law samples and scores in float64.
    Scores differentiate with respect to the original weights.

    Parameters
    ----------
    paths : _CappedPaths
        The compiled original graph, suffix, and bounded prefix plan.
    generator : torch.Generator
        The CPU generator for native Random draws.
    """

    def __init__(self, paths: _CappedPaths, *, generator: torch.Generator) -> None:
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("a capped path law needs a CPU generator")
        with torch.no_grad():
            paths.log_mass()
        self._paths = paths
        self._capacity = paths._capacity
        self._element = _capped_trace_type(self._capacity)
        self._generator = generator

    @property
    def element(self) -> TypeExpr:
        """Return the exact checked carrier of a draw.

        Returns
        -------
        TypeExpr
            The tagged original trace product.
        """
        return self._element

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw a retained path or the tail from the live full posterior.

        Returns
        -------
        tuple of torch.Tensor
            The CPU outcome flag, boundary states, and padded arc IDs.
        """
        with torch.no_grad():
            path = self._paths.sample(self._generator)
        if path is None:
            start, end, ids = -1, -1, (-1,) * self._capacity
        else:
            start, end = path.initial_state, path.final_state
            padding = (-1,) * (self._capacity - len(path.arc_ids))
            ids = (*path.arc_ids, *padding)
        return (
            torch.tensor(path is None, dtype=torch.bool),
            torch.tensor(start, dtype=torch.int64),
            torch.tensor(end, dtype=torch.int64),
            torch.tensor(ids, dtype=torch.int64),
        )

    def log_prob(self, value: object, keep_batch: bool = False) -> torch.Tensor:
        """Score one tagged outcome under the current pushforward law.

        Parameters
        ----------
        value : object
            The native tagged trace or tail record.
        keep_batch : bool
            The unbatched law accepts False.

        Returns
        -------
        torch.Tensor
            Its differentiable log mass, or log zero outside the support.

        Raises
        ------
        ValueError
            A batch or a value outside the typed carrier was supplied.
        PositionDependentControl
            A tag or occurrence identity depends on the evaluation position.
        """
        if keep_batch:
            raise ValueError("a capped path law has no batch axes")
        if not isinstance(value, tuple) or len(value) != 4:
            raise ValueError("a capped outcome needs a tag and three trace fields")
        tag, start, end, occurrences = value
        if (
            not isinstance(tag, torch.Tensor)
            or tag.ndim != 0
            or tag.dtype != torch.bool
        ):
            raise ValueError("a capped outcome needs a scalar Bool tag")
        initial = host_index(start, "path initial state")
        final = host_index(end, "path final state")
        if (
            not isinstance(occurrences, torch.Tensor)
            or occurrences.dtype != torch.int64
            or occurrences.shape != (self._capacity,)
        ):
            raise ValueError(
                "path arc occurrence tensor does not match its Int capacity"
            )
        if depends_on_position(tag) or depends_on_position(occurrences):
            raise PositionDependentControl("capped path outcome")
        ids = tuple(occurrences.tolist())
        if bool(tag):
            canonical = initial == final == -1 and all(arc == -1 for arc in ids)
            return self._paths.log_prob(None, tail=canonical)
        if -1 in ids:
            stop = ids.index(-1)
            if any(arc != -1 for arc in ids[stop:]):
                return self._paths.log_prob(None)
            ids = ids[:stop]
        return self._paths.log_prob((initial, ids, final))

    def log_mass(self, batch_axes: int = 0) -> torch.Tensor:
        """Return the log mass of the full tagged carrier.

        Parameters
        ----------
        batch_axes : int
            The retained batch rank, which must be zero.

        Returns
        -------
        torch.Tensor
            Log one after checking the live full path normalizer.

        Raises
        ------
        ValueError
            A batch axis was requested or the current normalizer is invalid.
        """
        if type(batch_axes) is not int or batch_axes != 0:
            raise ValueError("a capped path law has no batch axes")
        return self._paths.log_mass()


class FixedCappedCyclicRelationPathSampleRequest(QiecModel):
    """Request a checked cyclic path posterior with a single tail outcome.

    Parameters
    ----------
    request : FixedCyclicRelationPathSampleRequest
        Original nominal weights, ordered words, graph and local closure
        order limits, and the retained original trace capacity.
    max_closure_cells : int
        The aggregate local star cell limit, including auxiliary states.
    max_prefix_work : int
        The bound on charged prefix gathers, padded joins, layer values,
        and initial and final reductions.
    max_prefix_cells : int
        The bound on the number of active state and prefix depth pairs.
    """

    request: FixedCyclicRelationPathSampleRequest = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_closure_cells: int
    max_prefix_work: int
    max_prefix_cells: int

    @dx.model_validator()
    def _validate_limits(self) -> None:
        """Check the additional aggregate work and storage declarations.

        Raises
        ------
        ValueError
            A limit is outside the nonnegative Int range.
        """
        largest = torch.iinfo(torch.int64).max
        limits = (self.max_closure_cells, self.max_prefix_work, self.max_prefix_cells)
        if any(type(limit) is not int or not 0 <= limit <= largest for limit in limits):
            raise ValueError("capped path limits must be nonnegative Int values")

    @property
    def element_type(self) -> TypeExpr:
        """Return the distinct tagged outcome type.

        Returns
        -------
        TypeExpr
            The carrier of a retained trace or the tail.
        """
        return _capped_trace_type(self.request.request.trace_capacity)

    @property
    def result_type(self) -> TypeExpr:
        """Return the Sampleable type for capped paths.

        Returns
        -------
        TypeExpr
            The Sampleable instance over the distinct tagged carrier.
        """
        return sampleable_type(self.element_type)

    def bind(
        self, parameters: RelationWeightBindings, *, generator: torch.Generator
    ) -> CappedRelationPathLaw:
        """Bind the requested law to the original live weights.

        Parameters
        ----------
        parameters : RelationWeightBindings
            The original live scalar weight owners.
        generator : torch.Generator
            The CPU generator for native Random draws.

        Returns
        -------
        CappedRelationPathLaw
            The normalized law on retained traces and one tail value.

        Raises
        ------
        ValueError
            A binding, generator, or observed normalizer is invalid.
        RelationLimitError
            Graph or prefix construction exceeds a declared limit.
        """
        cyclic = self.request
        source = cyclic.request
        paths = _CappedPaths(
            source.relation.bind(parameters),
            source.words,
            source.trace_capacity,
            max_states=source.max_states,
            max_work=source.max_work,
            max_closure_size=cyclic.max_closure_size,
            max_closure_cells=self.max_closure_cells,
            max_prefix_work=self.max_prefix_work,
            max_prefix_cells=self.max_prefix_cells,
        )
        return CappedRelationPathLaw(paths, generator=generator)


__all__ = ["CappedRelationPathLaw", "FixedCappedCyclicRelationPathSampleRequest"]
