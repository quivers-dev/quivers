"""Reference-machine provider for serialized fixed-word relation requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

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
from quivers.qiec.relations import FixedRelationCostRequest, FixedRelationRequest
from quivers.qiec.types import UNIT, TypeExpr


_RELATION_SCORE_HANDLER_PREFIX = "__relation_score_"


def _relation_score_handler_id(module: str, name: str, operation: str) -> HandlerId:
    """Derive the checked handler identity of a source score.

    Parameters
    ----------
    module
        The QIEC module name.
    name
        The score's source name.
    operation
        The explicitly declared relation objective.

    Returns
    -------
    HandlerId
        The stable identity that includes the declared operation.
    """
    return HandlerId.derive(module, "relation_scores", name, operation, "handler")


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


class FixedRelationRuntimeProvider:
    """Bind one declared foreign Compute handler to a sparse request.

    The checked handler must have type `Compute[Unit, A]`, where `A`
    is the result type of the request, and resume once. Weight
    identities resolve to the caller's original tensors.

    Parameters
    ----------
    request : FixedRelationRequest or FixedRelationCostRequest
        The checked fixed observation, objective, and resource limits.
    parameters : Mapping[ParameterId, torch.Tensor]
        The live scalar tensors supplying the request's weight identities.
    handler_name : str
        The name of the foreign handler to bind in the checked module.
    """

    name = "sparse-relations"

    def __init__(
        self,
        request: FixedRelationRequest | FixedRelationCostRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        handler_name: str,
    ) -> None:
        if not isinstance(handler_name, str) or not handler_name:
            raise ValueError("a fixed relation provider requires a handler name")
        self._handler_name = handler_name
        self._operation = request.operation
        self._result_type = request.result_type
        relation = request.relation.bind(parameters)
        limits = dict(
            max_states=request.max_states,
            max_work=request.max_work,
            max_closure_size=request.max_closure_size,
        )
        self._score: Callable[[], torch.Tensor]
        if isinstance(request, FixedRelationCostRequest):
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
        request: FixedRelationRequest | FixedRelationCostRequest,
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
            The scalar score in its declared carrier with its gradient path.

        Raises
        ------
        TypeError
            The argument is outside the unit carrier.
        ValueError
            The score is outside the declared scalar carrier.
        """
        if argument is not None:
            raise TypeError("a fixed relation request takes a unit argument")
        score = self._score()
        if self._result_type == LOG_WEIGHT:
            result_algebra = LOG_PROB
        elif self._operation == "pair_weight":
            result_algebra = MARKOV
        else:
            result_algebra = REAL_ALGEBRA
        if (
            score.shape != ()
            or not score.is_floating_point()
            or not bool(result_algebra.contains(score))
        ):
            raise ValueError("fixed relation score is outside its scalar carrier")
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
                raise TypeError(
                    f"relation score {score_name!r} has a different declared operation "
                    f"from the supplied {self._operation!r} request"
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
        """Validate the request's differentiable scalar result.

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
