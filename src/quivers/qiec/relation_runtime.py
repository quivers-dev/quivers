"""Reference-machine provider for serialized fixed-word relation requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from hashlib import sha256

import torch

from quivers.core.algebras import LOG_PROB, MARKOV, REAL as REAL_ALGEBRA
from quivers.core.relations import FixedStringQuery
from quivers.qiec.builtins import COMPUTE, COMPUTE_APPLY, compute_handler
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.effects import EMPTY_ROW, ResumptionGrade
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.contracts import ParameterId
from quivers.qiec.identifiers import HandlerId
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_cascades import RelationCascadeSpec
from quivers.qiec.relation_composition import (
    ComposedRelationSpec,
    FixedComposedRelationRequest,
)
from quivers.qiec.relation_marginal_map import FixedRelationMarginalMapRequest
from quivers.qiec.relation_transforms import (
    BinaryRelationTransformSpec,
    FixedTransformedRelationRequest,
    UnaryRelationTransformSpec,
)
from quivers.qiec.relations import (
    FixedRelationArcCountsRequest,
    FixedRelationCostRequest,
    FixedRelationRequest,
    RationalRelationSpec,
)
from quivers.qiec.types import UNIT, TypeExpr


_RELATION_SCORE_HANDLER_PREFIX = "__relation_score_"


def _relation_score_handler_id(
    module: str,
    name: str,
    operation: str,
    relation: RationalRelationSpec
    | ComposedRelationSpec
    | UnaryRelationTransformSpec
    | BinaryRelationTransformSpec
    | RelationCascadeSpec
    | None = None,
) -> HandlerId:
    """Derive the checked handler identity of a source score.

    Parameters
    ----------
    module
        The QIEC module name.
    name
        The score's source name.
    operation
        The explicitly declared relation objective.
    relation
        An authored specification when the score binds its topology.

    Returns
    -------
    HandlerId
        The stable identity that includes the declared operation.
    """
    path = ("relation_scores", name, operation)
    if relation is not None:
        from quivers.qiec.serialization import dumps

        fingerprint = sha256(dumps(relation).encode("utf-8")).hexdigest()
        path += ("authored_topology", fingerprint)
    return HandlerId.derive(module, *path, "handler")


def _relation_score_handler_name(name: str) -> str:
    """Derive the handler name for a source-declared score.

    Parameters
    ----------
    name
        The score's source name.

    Returns
    -------
    str
        The generated handler's name.
    """
    return f"{_RELATION_SCORE_HANDLER_PREFIX}{name}"


def _expected_path_cost(
    query: FixedStringQuery,
    initial_costs: torch.Tensor | None,
    arc_costs: torch.Tensor | None,
    final_costs: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate the checked scalar expectation for the bound path costs.

    Parameters
    ----------
    query : FixedStringQuery
        The live fixed-word posterior query.
    initial_costs : torch.Tensor or None
        Costs on initial occurrences.
    arc_costs : torch.Tensor or None
        Costs on arc occurrences.
    final_costs : torch.Tensor or None
        Costs on final occurrences.

    Returns
    -------
    torch.Tensor
        The differentiable signed expected cost.
    """
    value = query.path_statistics(
        initial_costs=initial_costs, arc_costs=arc_costs, final_costs=final_costs
    ).expected_cost
    assert value is not None
    return value


def _selected_arc_counts(
    query: FixedStringQuery, arc_ids: tuple[int, ...]
) -> torch.Tensor:
    """Return expected traversal counts in request order.

    Parameters
    ----------
    query : FixedStringQuery
        The observed path graph with live original weight sources.
    arc_ids : tuple of int
        Original arc occurrence IDs in the requested result order.

    Returns
    -------
    torch.Tensor
        The selected real counts, using zero for absent occurrences.
    """
    result = query.expected_arc_counts()
    positions = {arc_id: row for row, arc_id in enumerate(result.arc_ids)}
    missing = len(result.arc_ids)
    rows = torch.tensor(
        [positions.get(arc_id, missing) for arc_id in arc_ids],
        dtype=torch.long,
        device=result.counts.device,
    )
    with_zero = torch.cat((result.counts, result.counts.new_zeros(1)))
    return with_zero.index_select(0, rows)


class FixedRelationRuntimeProvider:
    """Bind one declared foreign Compute handler to a sparse request.

    The checked handler must have type `Compute[Unit, A]`, where `A`
    is the result type of the request, and resume once. Weight
    identities resolve to the caller's original tensors.

    Parameters
    ----------
    request : FixedRelationRequest or FixedRelationCostRequest or FixedComposedRelationRequest or FixedRelationMarginalMapRequest or FixedRelationArcCountsRequest or FixedTransformedRelationRequest
        The checked fixed observation, objective, and resource limits.
    parameters : Mapping[ParameterId, torch.Tensor]
        The live scalar tensors supplying the request's weight identities.
    handler_name : str
        The name of the foreign handler to bind in the checked module.
    """

    name = "sparse-relations"

    def __init__(
        self,
        request: FixedRelationRequest
        | FixedRelationCostRequest
        | FixedRelationArcCountsRequest
        | FixedComposedRelationRequest
        | FixedRelationMarginalMapRequest
        | FixedTransformedRelationRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        handler_name: str,
    ) -> None:
        if not isinstance(handler_name, str) or not handler_name:
            raise ValueError("a fixed relation provider requires a handler name")
        self._handler_name = handler_name
        self._operation = request.operation
        self._relation_spec = request.relation
        self._result_type = request.result_type
        self._result_shape = (
            (len(request.arc_ids),)
            if isinstance(request, FixedRelationArcCountsRequest)
            else ()
        )
        self._score: Callable[[], torch.Tensor]
        if isinstance(
            request, FixedComposedRelationRequest | FixedTransformedRelationRequest
        ):
            self._score = request.plan(parameters).weight
            return
        relation = request.relation.bind(parameters)
        if isinstance(request, FixedRelationMarginalMapRequest):
            zero = relation._weights().new_full((), -torch.inf)

            def marginal_map_score() -> torch.Tensor:
                """Read the live score of the current winning output.

                Returns
                -------
                torch.Tensor
                    The selected log pair weight or the LogProb zero.
                """
                choice = request._choose(relation)
                return zero if choice is None else choice.log_pair_weight

            self._score = marginal_map_score
            return
        limits = dict(
            max_states=request.max_states,
            max_work=request.max_work,
            max_closure_size=request.max_closure_size,
        )
        if isinstance(request, FixedRelationArcCountsRequest):
            query = relation.query(request.words, **limits)
            self._score = partial(_selected_arc_counts, query, request.arc_ids)
        elif isinstance(request, FixedRelationCostRequest):
            query = relation.query(request.words, **limits)
            anchor = relation._weights()
            initial, arcs, final = (
                None if group is None else anchor.new_tensor(group)
                for group in (
                    request.initial_costs,
                    request.arc_costs,
                    request.final_costs,
                )
            )
            self._score = partial(_expected_path_cost, query, initial, arcs, final)
        elif request.operation == "pair_weight":
            self._score = relation.query(request.words, **limits).weight
        elif request.operation == "best_path_weight":
            self._score = relation.query(request.words, **limits).best_path_weight
        elif request.operation == "path_entropy":
            query = relation.query(request.words, **limits)
            self._score = lambda: query.path_statistics().entropy
        else:
            split = relation.topology.input_tapes
            conditional = relation.conditional(
                request.words[:split],
                max_states=request.max_states,
                max_work=request.max_work,
                max_closure_size=request.max_closure_size,
            )
            self._score = lambda: conditional.log_prob(request.words[split:])

    @classmethod
    def for_score(
        cls,
        name: str,
        request: FixedRelationRequest
        | FixedRelationCostRequest
        | FixedRelationArcCountsRequest
        | FixedComposedRelationRequest
        | FixedRelationMarginalMapRequest
        | FixedTransformedRelationRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> FixedRelationRuntimeProvider:
        """Bind a fixed request to a source-declared relation score.

        Parameters
        ----------
        name
            The name used in the source declaration.
        request
            The serialized and validated fixed relation request.
        parameters
            The original scalar parameter objects indexed by identity.

        Returns
        -------
        FixedRelationRuntimeProvider
            A provider for the generated checked handler.

        Raises
        ------
        ValueError
            If the source name is empty or the binding is invalid.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("a source-declared relation score requires a name")
        return cls(request, parameters, handler_name=_relation_score_handler_name(name))

    def _evaluate(self, argument: object, _context: ClauseContext) -> torch.Tensor:
        """Evaluate the objective using its current parameter values.

        Parameters
        ----------
        argument : object
            The evaluated unit argument to `Compute`.
        _context : ClauseContext
            The clause context supplied by the reference machine.

        Returns
        -------
        torch.Tensor
            The query result in its declared carrier with its gradient path.

        Raises
        ------
        TypeError
            The argument is outside the unit carrier.
        ValueError
            The result is outside its declared carrier.
        """
        if argument is not None:
            raise TypeError("a fixed relation request takes a unit argument")
        score = self._score()
        if self._result_type == LOG_WEIGHT:
            result_algebra = LOG_PROB
        elif self._operation in (
            "pair_weight",
            "best_path_weight",
            "expected_arc_counts",
        ):
            result_algebra = MARKOV
        else:
            result_algebra = REAL_ALGEBRA
        if (
            score.shape != self._result_shape
            or not score.is_floating_point()
            or not bool(result_algebra.contains(score).all())
        ):
            raise ValueError("fixed relation score is outside its declared carrier")
        return score

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach the provider to its exact checked foreign handler.

        Parameters
        ----------
        module : QiecModule
            The checked module declaring the handler.
        attachments : RuntimeAttachments
            The runtime bindings for the invocation.

        Raises
        ------
        ValueError
            The module has no unique handler with the requested name.
        TypeError
            The declaration does not have the required linear compute contract.
        """
        declarations = [
            handler for handler in module.handlers if handler.name == self._handler_name
        ]
        if len(declarations) != 1:
            raise ValueError(f"unknown fixed relation handler {self._handler_name!r}")
        definition = declarations[0]
        result = self._result_type
        expected = COMPUTE.with_(arguments=(UNIT, result))
        clauses = tuple(
            (clause.operation, clause.grade) for clause in definition.clauses
        )
        if (
            definition.implementation != "foreign"
            or definition.effect != expected
            or definition.input_type != result
            or definition.output_type != result
            or clauses != ((COMPUTE_APPLY, ResumptionGrade.LINEAR),)
            or definition.parameters
            or definition.telescope
            or definition.introduced != EMPTY_ROW
            or not definition.total
            or definition.forwards_unknown
            or definition.return_clause is not None
            or definition.foreign is not None
        ):
            raise TypeError(
                f"incompatible fixed relation handler {self._handler_name!r}"
            )
        score_name = definition.name.removeprefix(_RELATION_SCORE_HANDLER_PREFIX)
        if score_name != definition.name:
            expected_id = _relation_score_handler_id(
                module.module, score_name, self._operation
            )
            if definition.id != expected_id:
                bound_id = _relation_score_handler_id(
                    module.module, score_name, self._operation, self._relation_spec
                )
                if definition.id != bound_id:
                    raise TypeError(
                        f"relation score {score_name!r} has a different declared operation "
                        f"or authored topology from the supplied {self._operation!r} request"
                    )
        validation = self.validator_for(result)
        assert validation is not None
        runtime = compute_handler(
            self._evaluate,
            effect=expected,
            result_validator=validation,
            answer_type=result,
            key=self._handler_name,
        )
        attachments.bind_handler(runtime.with_(definition=definition))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Validate the request's differentiable result.

        Parameters
        ----------
        type_ : TypeExpr
            The requested checked result type.

        Returns
        -------
        RuntimeValidator or None
            The predicate for this request's result type, or `None` for other types.
        """
        if type_ != self._result_type:
            return None
        core = CoreRuntimeProvider().validator_for(type_)
        if self._result_shape:
            return core
        algebra = LOG_PROB if type_ == LOG_WEIGHT else None

        def validate(value: object) -> bool:
            """Check a value against the declared scalar carrier.

            Parameters
            ----------
            value : object
                The runtime value to check.

            Returns
            -------
            bool
                Whether the value inhabits the requested type.
            """
            if core is not None and core(value):
                return True
            if not (
                isinstance(value, torch.Tensor)
                and value.shape == ()
                and value.is_floating_point()
            ):
                return False
            return bool(
                torch.isfinite(value) if algebra is None else algebra.contains(value)
            )

        return validate


__all__ = ["FixedRelationRuntimeProvider"]
