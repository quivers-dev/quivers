"""Reference-machine provider for serialized fixed-word relation requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.qiec.builtins import COMPUTE, COMPUTE_APPLY, compute_handler
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.effects import EMPTY_ROW, ResumptionGrade
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.contracts import ParameterId
from quivers.qiec.module import QiecModule
from quivers.qiec.relations import FixedRelationRequest
from quivers.qiec.types import UNIT, TypeExpr


class FixedRelationRuntimeProvider:
    """Bind one declared foreign Compute handler to a sparse request.

    The checked handler must have type `Compute[Unit, A]`, where `A`
    is the result type of the request, and resume once. Weight
    identities resolve to the caller's original tensors.

    Parameters
    ----------
    request : FixedRelationRequest
        The checked fixed observation, objective, and resource limits.
    parameters : Mapping[ParameterId, torch.Tensor]
        The live scalar tensors supplying the request's weight identities.
    handler_name : str
        The name of the foreign handler to bind in the checked module.
    """

    name = "sparse-relations"

    def __init__(
        self,
        request: FixedRelationRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        handler_name: str,
    ) -> None:
        if not isinstance(handler_name, str) or not handler_name:
            raise ValueError("a fixed relation provider requires a handler name")
        self._handler_name = handler_name
        self._result_type = request.result_type
        relation = request.relation.bind(parameters)
        limits = dict(
            max_states=request.max_states,
            max_work=request.max_work,
            max_closure_size=request.max_closure_size,
        )
        self._score: Callable[[], torch.Tensor]
        if request.operation == "pair_weight":
            self._score = relation.query(request.words, **limits).weight
        else:
            split = relation.topology.input_tapes
            conditional = relation.conditional(
                request.words[:split],
                max_states=request.max_states,
                max_work=request.max_work,
                max_closure_size=request.max_closure_size,
            )
            self._score = lambda: conditional.log_prob(request.words[split:])

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
        result_algebra = LOG_PROB if self._result_type == LOG_WEIGHT else MARKOV
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
