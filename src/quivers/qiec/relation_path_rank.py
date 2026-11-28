"""Checked deterministic ranking of original sparse path occurrences."""

from __future__ import annotations

from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    ProductiveEpsilonCycleError,
    RelationLimitError,
    Words,
)
from quivers.core.relations.paths import _PathWalks
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_path_law import _check_trace_capacity
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import (
    RationalRelationSpec,
    RelationWeightBindings,
    _validate_fixed_observation,
)
from quivers.qiec.types import INT, REAL, IndexLiteral, TypeExpr, product_type

type _RankedPaths = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class FixedRelationPathRankRequest(QiecModel):
    """Rank positive original paths on a fixed-word productive DAG.

    The result contains a valid-row count, a pair of original boundary
    states per row, ordered original arc IDs, and unnormalized path
    weights. Trailing arc positions and missing rows use -1. Missing
    weights use the semiring zero. The first row with rank count one
    is the ordinary deterministic best path.

    Parameters
    ----------
    relation : RationalRelationSpec
        The typed Markov or LogProb topology and original weight owners.
    words : Words
        One observed word per tape in tape order.
    k : int
        Positive number of result rows and maximum retained path rank.
    trace_capacity : int
        Extent of each row's original arc occurrence vector.
    max_states, max_work : int
        Limits on states and candidate arcs in the fixed-word graph.
    max_rank_entries : int
        Bound on the product of k and one plus the active state count.
    max_rank_work : int
        Bound on weight-slot reads, stream visits, and the heap work allowance.
    max_result_cells : int
        Total scalar-cell limit across the result's four fields.
    """

    relation: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    k: int
    trace_capacity: int
    max_states: int
    max_work: int
    max_rank_entries: int
    max_rank_work: int
    max_result_cells: int

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the observation, result shape, and static work limits.

        Raises
        ------
        ValueError
            Invalid words, limits, rank, or integer boundary range.
        RelationLimitError
            The declared result shape exceeds its scalar-cell limit.
        """
        _validate_fixed_observation(
            self.relation, self.words, self.max_states, self.max_work, None
        )
        _check_trace_capacity(self.relation.topology, self.trace_capacity)
        if type(self.k) is not int or self.k < 1:
            raise ValueError("path rank k must be a positive integer")
        for name, limit in (
            ("max_states", self.max_states),
            ("max_work", self.max_work),
            ("max_rank_entries", self.max_rank_entries),
            ("max_rank_work", self.max_rank_work),
            ("max_result_cells", self.max_result_cells),
        ):
            if type(limit) is not int or limit < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        result_cells = 1 + self.k * (self.trace_capacity + 3)
        if result_cells > self.max_result_cells:
            raise RelationLimitError("ranked paths exceed max_result_cells")
        if result_cells > torch.iinfo(torch.int64).max:
            raise ValueError("ranked path result dimensions exceed the Int carrier")

    @property
    def handler_name(self) -> str:
        """Derive the checked owner of the observation and rank shape.

        Returns
        -------
        str
            The foreign handler identity of the topology, words and shape.
        """
        from quivers.qiec.serialization import dumps

        source = self.relation, self.words, self.k, self.trace_capacity
        owner = sha256(dumps(source).encode("utf-8")).hexdigest()
        return f"__relation_path_rank_{owner}"

    @property
    def result_type(self) -> TypeExpr:
        """Type the count, endpoints, original arc IDs, and scores.

        Returns
        -------
        TypeExpr
            A scalar Int, two Int matrices, and an algebra-specific vector.
        """
        rows = IndexLiteral(value=self.k, sort=NAT)
        pair = IndexLiteral(value=2, sort=NAT)
        capacity = IndexLiteral(value=self.trace_capacity, sort=NAT)
        score = LOG_WEIGHT if self.relation.algebra == LOG_PROB.subject else REAL
        return product_type(
            INT,
            tensor_type(INT, (rows, pair)),
            tensor_type(INT, (rows, capacity)),
            tensor_type(score, (rows,)),
        )


class FixedRelationPathRankRuntimeProvider:
    """Bind a deterministic sparse path ranking to checked native Compute.

    The provider compiles the observation graph once. Each evaluation
    reads the current weights and ranks complete original paths. A
    returned score differentiates its selected path with the occurrence
    IDs fixed, including at ties in the deterministic ordering.

    Parameters
    ----------
    request : FixedRelationPathRankRequest
        The checked topology, observation, exact rank, and limits.
    parameters : RelationWeightBindings
        The original live weight owners of the relation.

    Raises
    ------
    ProductiveEpsilonCycleError
        The structural observation graph has a productive epsilon cycle.
    RelationLimitError
        Observation construction, rank storage, or rank work exceeds a limit.
    """

    name = "fixed-relation-path-rank"

    def __init__(
        self, request: FixedRelationPathRankRequest, parameters: RelationWeightBindings
    ) -> None:
        self._request = request
        self._walks = _PathWalks(
            request.relation.topology,
            request.words,
            request.max_states,
            request.max_work,
            None,
        )
        if self._walks._cyclic:
            raise ProductiveEpsilonCycleError(
                "path ranking requires an acyclic productive graph"
            )
        if request.k * (len(self._walks._order) + 1) > request.max_rank_entries:
            raise RelationLimitError("ranked paths exceed max_rank_entries")
        degrees = (
            len(self._walks._initial),
            *(
                len(self._walks._arcs[state]) + int(state in self._walks._final)
                for state in self._walks._order
            ),
        )
        # Each merge visits d structural streams and uses at most k rank
        # positions with an O(1 + log(d + 1)) heap operation per position.
        rank_work = len(request.relation.topology.weight_names) + sum(
            d + request.k * (1 + d.bit_length()) for d in degrees if d
        )
        if rank_work > request.max_rank_work:
            raise RelationLimitError("ranked paths exceed max_rank_work")
        self._relation = request.relation.bind(parameters)
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    def _evaluate(self, argument: object, _context: ClauseContext) -> _RankedPaths:
        """Pack complete ranked walks and their differentiable weights.

        Parameters
        ----------
        argument : object
            The Unit argument supplied to Compute.
        _context : ClauseContext
            The checked context of the calling computation.

        Returns
        -------
        tuple of torch.Tensor
            Path count, original endpoint states, original arcs, and weights.

        Raises
        ------
        TypeError
            The supplied argument does not inhabit Unit.
        RelationLimitError
            A selected path does not fit the complete trace carrier.
        """
        if argument is not None:
            raise TypeError("sparse path ranking takes a unit argument")
        request = self._request
        weights = self._relation._weights()
        algebra = self._relation.algebra
        paths = self._walks.k_best(request.k, weights, algebra)
        if any(len(path.arc_ids) > request.trace_capacity for path in paths):
            raise RelationLimitError("selected path exceeds trace_capacity")
        count = len(paths)
        missing = request.k - count
        boundaries = [(path.initial_state, path.final_state) for path in paths]
        boundaries.extend([(-1, -1)] * missing)
        padding = (-1,) * request.trace_capacity
        arcs = [path.arc_ids + padding[len(path.arc_ids) :] for path in paths]
        arcs.extend([padding] * missing)
        zeros = algebra.zero_like((missing,), weights)
        scores = (
            torch.cat((torch.stack([p.weight for p in paths]), zeros))
            if paths
            else zeros
        )
        return (
            weights.new_tensor(count, dtype=torch.int64),
            weights.new_tensor(boundaries, dtype=torch.int64),
            weights.new_tensor(arcs, dtype=torch.int64),
            scores,
        )

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Bind the provider to the matching authored foreign Compute.

        Parameters
        ----------
        module : QiecModule
            Module with the checked once-resuming handler.
        attachments : RuntimeAttachments
            Runtime handler bindings for the computation.

        Raises
        ------
        ValueError
            The module has no handler for this exact observation and shape.
        TypeError
            The matched handler has an incompatible Compute contract.
        """
        definition = _fixed_compute_definition(
            module, self._handler_name, self._result_type
        )
        validator = self.validator_for(self._result_type)
        assert validator is not None
        runtime = compute_handler(
            self._evaluate,
            effect=definition.effect,
            result_validator=validator,
            answer_type=definition.output_type,
            key=definition.name,
        )
        attachments.bind_handler(runtime.with_(definition=definition))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Supply the core validator for this exact numeric product.

        Parameters
        ----------
        type_ : TypeExpr
            The result carrier requested by the reference machine.

        Returns
        -------
        RuntimeValidator or None
            The checked product validator, when the result carrier matches.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = ["FixedRelationPathRankRequest", "FixedRelationPathRankRuntimeProvider"]
