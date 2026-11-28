"""Reference-machine bindings for checked exact string graph queries."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.qiec.builtins import COMPUTE, COMPUTE_APPLY, compute_handler
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.effects import HandlerClauseDef, HandlerDef, ResumptionGrade
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.identifiers import HandlerId
from quivers.qiec.module import QiecModule
from quivers.qiec.string_graphs import TreeStringGraphRequest
from quivers.qiec.terms import Local
from quivers.qiec.types import UNIT, TypeExpr


def _string_graph_score_handler_name(name: str) -> str:
    """Return the generated handler name of a source graph score.

    Parameters
    ----------
    name
        The score name in QVR source.

    Returns
    -------
    str
        The checked handler name.
    """
    return f"__string_graph_score_{name}"


def _tree_string_graph_handler(
    module: str, name: str, operation: str, result: TypeExpr
) -> HandlerDef:
    """Build the shared checked handler declaration.

    Parameters
    ----------
    module : str
        The checked module name.
    name : str
        The handler name in the module.
    operation : str
        The exact graph objective that the handler serves.
    result : TypeExpr
        The validated closed scalar result type.

    Returns
    -------
    HandlerDef
        The linear foreign Compute handler.
    """
    effect = COMPUTE.with_(arguments=(UNIT, result))
    return HandlerDef(
        name=name,
        id=HandlerId.derive(module, "string_graph", name, operation),
        effect=effect,
        clauses=(
            HandlerClauseDef(
                operation=COMPUTE_APPLY,
                grade=ResumptionGrade.LINEAR,
                parameters=(Local(name="argument", type=UNIT),),
            ),
        ),
        input_type=result,
        output_type=result,
    )


def tree_string_graph_handler(
    module: str, name: str, request: TreeStringGraphRequest
) -> HandlerDef:
    """Declare a linear unit-argument handler for a checked graph query.

    The stable identity includes the graph query's objective. A log
    partition and normalized log probability have distinct handler
    identities while inhabiting the same LogWeight result type.

    Parameters
    ----------
    module : str
        The checked module name.
    name : str
        The handler name in the module.
    request : TreeStringGraphRequest
        The operation and result type to serve.

    Returns
    -------
    HandlerDef
        The checked foreign Compute declaration.

    Raises
    ------
    ValueError
        If the module or handler name is empty.
    """
    if not module or not name:
        raise ValueError(
            "a tree string graph handler requires module and handler names"
        )
    return _tree_string_graph_handler(
        module, name, request.operation, request.result_type
    )


class TreeStringGraphRuntimeProvider:
    """Serve an exact graph objective through its checked Compute declaration.

    Parameters
    ----------
    request : TreeStringGraphRequest
        The graph, query objective, and resource limits.
    parameters : Mapping of ParameterId to torch.Tensor
        The caller's original scalar leaves supplying the graph weights.
    handler_name : str
        The name of the exact handler declared in the checked module.
    """

    name = "tree-string-graphs"

    def __init__(
        self,
        request: TreeStringGraphRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        handler_name: str,
    ) -> None:
        if not isinstance(handler_name, str) or not handler_name:
            raise ValueError("a tree string graph provider requires a handler name")
        self._request = request
        self._handler_name = handler_name
        self._score = request.bind(parameters)

    @classmethod
    def for_score(
        cls,
        name: str,
        request: TreeStringGraphRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> TreeStringGraphRuntimeProvider:
        """Bind the request to a named QVR string graph score.

        Parameters
        ----------
        name
            The name declared in QVR source.
        request
            The checked graph objective.
        parameters
            The original scalar tensors indexed by parameter identity.

        Returns
        -------
        TreeStringGraphRuntimeProvider
            The provider for the source-generated handler.

        Raises
        ------
        ValueError
            If the score name is empty or the request cannot be bound.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("a source graph score requires a name")
        return cls(
            request, parameters, handler_name=_string_graph_score_handler_name(name)
        )

    def _evaluate(self, argument: object, _context: ClauseContext) -> torch.Tensor:
        """Evaluate the bound graph with live weights.

        Parameters
        ----------
        argument
            The unit argument of the checked Compute operation.
        _context : ClauseContext
            The reference machine's handler context.

        Returns
        -------
        torch.Tensor
            The checked graph score and its gradient path.

        Raises
        ------
        TypeError
            If the argument is outside the unit carrier.
        ValueError
            If the score is outside the expected scalar carrier.
        """
        if argument is not None:
            raise TypeError("a tree string graph request takes a unit argument")
        score = self._score()
        algebra = LOG_PROB if self._request.result_type == LOG_WEIGHT else MARKOV
        if (
            score.shape != ()
            or not score.is_floating_point()
            or not bool(algebra.contains(score))
        ):
            raise ValueError("tree string graph score is outside its scalar carrier")
        return score

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Install the provider for the exact declared operation.

        Parameters
        ----------
        module : QiecModule
            The checked module declaring the foreign handler.
        attachments : RuntimeAttachments
            The invocation's runtime handler table.

        Raises
        ------
        TypeError
            If the checked handler does not declare this objective and type.
        """
        expected = tree_string_graph_handler(
            module.module, self._handler_name, self._request
        )
        if expected not in module.handlers:
            raise TypeError(
                f"incompatible tree string graph handler {self._handler_name!r}"
            )
        validator = self.validator_for(self._request.result_type)
        assert validator is not None
        runtime = compute_handler(
            self._evaluate,
            effect=expected.effect,
            result_validator=validator,
            answer_type=expected.output_type,
            key=self._handler_name,
        )
        attachments.bind_handler(runtime.with_(definition=expected))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Validate torch scalars inhabiting the declared result type.

        Parameters
        ----------
        type_ : TypeExpr
            The closed QIEC type requested by the runtime.

        Returns
        -------
        RuntimeValidator or None
            The scalar predicate for this result type, or None for other types.
        """
        if type_ != self._request.result_type:
            return None
        core = CoreRuntimeProvider().validator_for(type_)

        def validate(value: object) -> bool:
            """Check a host value against the expected scalar carrier.

            Parameters
            ----------
            value
                The host value.

            Returns
            -------
            bool
                Whether the value inhabits the requested type.
            """
            if core is not None and core(value):
                return True
            if not isinstance(value, torch.Tensor) or (
                value.shape != () or not value.is_floating_point()
            ):
                return False
            return bool(
                LOG_PROB.contains(value)
                if type_ == LOG_WEIGHT
                else torch.isfinite(value)
            )

        return validate


__all__ = ["TreeStringGraphRuntimeProvider", "tree_string_graph_handler"]
