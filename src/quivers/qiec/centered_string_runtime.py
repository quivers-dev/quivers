"""Checked native reports for centered acyclic string probabilities."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import CenteredStringLogScore
from quivers.core.relations._centered import _CenteredBinding
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import FixedRelationRequest, RelationWeightBindings
from quivers.qiec.string_graphs import TreeStringGraphRequest
from quivers.qiec.types import INT, IndexLiteral, TypeExpr, product_type


class CenteredStringLogProbRequest(QiecModel):
    """Request a bounded centered native LogProb likelihood.

    Parameters
    ----------
    observation : FixedRelationRequest or TreeStringGraphRequest
        A checked fixed-input conditional or fixed-word tree marginal.
        It must declare a LogProb model without a productive closure.
    max_center_occurrences : int
        The combined cap for productive arcs and boundary marks across
        the numerator and denominator.
    max_dyadic_work : int
        The combined upper limit on exact-pass slot, state, and
        occurrence visits.
    """

    observation: FixedRelationRequest | TreeStringGraphRequest = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_center_occurrences: int
    max_dyadic_work: int

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the supported objectives, precision scope and work limits.

        Raises
        ------
        ValueError
            The objective or one of its policies is outside the supported domain.
        """
        observation = self.observation
        if not isinstance(observation, (FixedRelationRequest, TreeStringGraphRequest)):
            raise TypeError("a centered native request requires a checked observation")
        if isinstance(observation, FixedRelationRequest):
            objective = observation.operation == "conditional_log_prob"
            algebra = observation.relation.algebra
        else:
            objective = observation.word is not None
            algebra = observation.graph.algebra
        if (
            not objective
            or algebra != LOG_PROB.subject
            or observation.max_closure_size is not None
        ):
            raise ValueError(
                "a centered native request requires an acyclic LogProb "
                "conditional or tree word probability"
            )
        for name, limit in (
            ("max_center_occurrences", self.max_center_occurrences),
            ("max_dyadic_work", self.max_dyadic_work),
        ):
            if type(limit) is not int or not 0 <= limit <= torch.iinfo(torch.int64).max:
                raise ValueError(f"{name} must be a nonnegative Int")

    @property
    def handler_name(self) -> str:
        """Identify the nominal graph, objective and words of this request.

        Returns
        -------
        str
            The checked handler name shared by resource policies for the same objective.
        """
        from quivers.qiec.serialization import dumps

        observation = self.observation
        if isinstance(observation, FixedRelationRequest):
            identity = ("conditional", observation.relation, observation.words)
        else:
            identity = (
                "tree",
                observation.graph,
                observation.variable,
                observation.word,
            )
        digest = sha256(dumps(identity).encode("utf-8")).hexdigest()
        return f"__centered_string_log_prob_{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Return the checked type for the log score and measured work.

        Returns
        -------
        TypeExpr
            A LogWeight and the four state, occurrence, slot and exact-pass counts.
        """
        work = tensor_type(INT, (IndexLiteral(value=4, sort=NAT),))
        return product_type(LOG_WEIGHT, work)

    def _compile(
        self, parameters: RelationWeightBindings
    ) -> Callable[[], CenteredStringLogScore]:
        """Bind the original weights and compile the declared observation.

        Parameters
        ----------
        parameters : RelationWeightBindings
            The live scalar owners expected by the checked model.

        Returns
        -------
        Callable
            The reusable centered query over these original weight owners.
        """
        plan, weights = self._binding(parameters)

        def evaluate() -> CenteredStringLogScore:
            """Evaluate the centered score over the current original weights.

            Returns
            -------
            CenteredStringLogScore
                The live score and its measured chart work.
            """
            return plan.evaluate(weights())

        return evaluate

    def _binding(self, parameters: RelationWeightBindings) -> _CenteredBinding:
        """Bind the reusable plan and its original weight reader.

        Parameters
        ----------
        parameters : RelationWeightBindings
            The live scalar owners expected by the checked model.

        Returns
        -------
        tuple
            The centered query plan and its evaluated weight source.
        """
        observation = self.observation
        limits = {
            "max_states": observation.max_states,
            "max_work": observation.max_work,
        }
        centering = {
            "max_center_occurrences": self.max_center_occurrences,
            "max_dyadic_work": self.max_dyadic_work,
        }
        if isinstance(observation, FixedRelationRequest):
            relation = observation.relation.bind(parameters)
            split = relation.topology.input_tapes
            conditional = relation.conditional(observation.words[:split], **limits)
            return conditional._centered_binding(observation.words[split:], **centering)
        graph = observation.graph.bind(parameters, **limits)
        marginal = graph.tree_marginal(observation.variable, **limits)
        word = observation.word
        assert word is not None
        return marginal._centered_binding(word, **centering)


class CenteredStringLogProbRuntimeProvider:
    """Bind one centered native report to its original live weight owners.

    Parameters
    ----------
    request : CenteredStringLogProbRequest
        The fixed objective, typed model and compilation limits.
    parameters : RelationWeightBindings
        The live scalar owners expected by the checked relation or graph.
    """

    name = "centered-string-log-prob"

    def __init__(
        self, request: CenteredStringLogProbRequest, parameters: RelationWeightBindings
    ) -> None:
        self._query = request._compile(parameters)
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    def _evaluate(
        self, argument: object, _context: ClauseContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the named Compute request with the current original weights.

        Parameters
        ----------
        argument : object
            The Unit argument for the fixed observation.
        _context : ClauseContext
            The checked handler's reference execution context.

        Returns
        -------
        tuple
            The live centered log probability and measured work.

        Raises
        ------
        TypeError
            The argument does not inhabit Unit.
        """
        if argument is not None:
            raise TypeError("a centered string query takes a unit argument")
        result = self._query()
        work = result.log_prob.new_tensor(
            (
                result.states,
                result.occurrences,
                result.weight_slots,
                result.dyadic_work,
            ),
            dtype=torch.int64,
        )
        return result.log_prob, work

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Install the runtime for this exact checked Compute declaration.

        Parameters
        ----------
        module : QiecModule
            The module containing the named foreign handler.
        attachments : RuntimeAttachments
            The checked invocation's attachment table.

        Raises
        ------
        TypeError
            The corresponding declaration does not have the checked contract.
        ValueError
            The module lacks the expected foreign handler.
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
        """Supply the runtime predicate for the checked report.

        Parameters
        ----------
        type_ : TypeExpr
            The requested runtime result type.

        Returns
        -------
        RuntimeValidator or None
            The canonical validator for the matching report type.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = ["CenteredStringLogProbRequest", "CenteredStringLogProbRuntimeProvider"]
