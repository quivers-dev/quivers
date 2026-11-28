"""Expose a bounded regular string EP report to checked QIEC source."""

from __future__ import annotations

from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.relations import RelationLimitError
from quivers.core.relations.regular_projection import _RegularZeroMassError
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import RelationWeightBindings
from quivers.qiec.string_graphs import StringFactorGraphSpec
from quivers.qiec.string_solvers import RegularStringEPConfig, RegularStringEPPolicy
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexLiteral,
    TypeExpr,
    product_type,
)

type _Report = tuple[
    str,
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class RegularStringEPRequest(QiecModel):
    """Query finitely many words in one variable's fitted regular belief.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The exact typed source graph and original parameter owners.
    config : RegularStringEPConfig
        The reference languages, feature families, and declared schedule.
    variable : str
        The variable whose normalized log probabilities are requested.
    words : tuple[tuple[int, ...], ...]
        Distinct probes in their requested order. Words outside finite
        reference support have zero probability.
    """

    graph: StringFactorGraphSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    config: RegularStringEPConfig = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    variable: str
    words: tuple[tuple[int, ...], ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _check_query(self) -> None:
        """Check that the references and probes use the exact nominal tapes.

        Raises
        ------
        ValueError
            If the graph variable, bases, or ordered probe words are invalid.
        """
        alphabets = {
            name: (alphabet, size) for name, alphabet, size, _ in self.graph.variables
        }
        references = dict(self.config.bases)
        if (
            self.variable not in alphabets
            or alphabets.keys() != references.keys()
            or any(
                spec.tapes != (alphabets[name],) for name, spec in references.items()
            )
        ):
            raise ValueError(
                "regular EP references must match the graph's nominal tapes"
            )
        _, size = alphabets[self.variable]
        if (
            not self.words
            or any(
                not isinstance(word, tuple)
                or any(
                    type(symbol) is not int or not 0 <= symbol < size for symbol in word
                )
                for word in self.words
            )
            or len(set(self.words)) != len(self.words)
        ):
            raise ValueError("regular EP requires distinct words in the query alphabet")

    @property
    def handler_name(self) -> str:
        """Bind the checked graph, reference family, variable, and word order.

        Returns
        -------
        str
            The canonical foreign handler name for those static inputs.
        """
        from quivers.qiec.serialization import dumps

        static = (
            self.graph,
            self.config.bases,
            self.config.features,
            self.variable,
            self.words,
        )
        identity = sha256(dumps(static).encode("utf-8")).hexdigest()
        return f"__string_graph_regular_ep_{identity}"

    @property
    def result_type(self) -> TypeExpr:
        """Give the typed report's result carrier.

        Returns
        -------
        TypeExpr
            Status, reason, accepted sweeps, residual presence and value,
            belief presence, log probabilities, and attempted local fits.
        """
        size = IndexLiteral(value=len(self.words), sort=NAT)
        return product_type(
            STRING, STRING, INT, BOOL, REAL, BOOL, tensor_type(LOG_WEIGHT, (size,)), INT
        )


class RegularStringEPRuntimeProvider:
    """Run an authored report with the existing bounded regular EP executor.

    Parameters
    ----------
    request : RegularStringEPRequest
        The checked graph, references, and solver settings.
    parameters : RelationWeightBindings
        The original live weight owners. Each execution snapshots their
        current values. The fitted report has no derivative through EP.
    policy : RegularStringEPPolicy or None
        Optional optimizer and initialization policy wrapping the exact
        request configuration. None uses the ordinary unit initialization
        and adaptive backtracking.

    Raises
    ------
    ValueError
        If the supplied policy wraps a different solver configuration.
    """

    name = "regular-string-ep"

    def __init__(
        self,
        request: RegularStringEPRequest,
        parameters: RelationWeightBindings,
        *,
        policy: RegularStringEPPolicy | None = None,
    ) -> None:
        if policy is not None and policy.config != request.config:
            raise ValueError("regular EP policy must wrap the exact request config")
        self._request = request
        self._parameters = parameters
        self._solver = request.config if policy is None else policy
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Return the measured status and any fitted word probabilities.

        Parameters
        ----------
        argument : object
            The unit argument to Compute.
        _context : ClauseContext
            The checked handler context.

        Returns
        -------
        tuple
            The report in its declared coordinate order. A failed local
            projection retains the last accepted belief. Resource failure or
            an evaluated log-zero normalizer leaves the report unavailable.

        Raises
        ------
        TypeError
            If the argument is outside Unit.
        """
        if argument is not None:
            raise TypeError("a regular EP report takes a unit argument")
        request = self._request
        try:
            result = self._solver.infer(request.graph, self._parameters)
            belief = result.marginals[request.variable]
            with torch.no_grad():
                scores = torch.stack(
                    tuple(belief.log_prob(word) for word in request.words)
                )
        except (RelationLimitError, _RegularZeroMassError) as error:
            status = (
                "resource_limit"
                if isinstance(error, RelationLimitError)
                else "zero_mass"
            )
            missing = torch.full((len(request.words),), -torch.inf, dtype=torch.float64)
            return (
                status,
                str(error),
                missing.new_zeros((), dtype=torch.int64),
                missing.new_tensor(False, dtype=torch.bool),
                missing.new_zeros(()),
                missing.new_tensor(False, dtype=torch.bool),
                missing,
                missing.new_zeros((), dtype=torch.int64),
            )
        residual = result.message_residual
        return (
            result.status,
            result.updates[-1].fit.status
            if result.status == "projection_failed"
            else "",
            scores.new_tensor(result.sweeps, dtype=torch.int64),
            scores.new_tensor(residual is not None, dtype=torch.bool),
            scores.new_tensor(0.0 if residual is None else residual),
            scores.new_tensor(True, dtype=torch.bool),
            scores,
            scores.new_tensor(len(result.updates), dtype=torch.int64),
        )

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach only to the exact authored foreign Compute declaration.

        Parameters
        ----------
        module : QiecModule
            The checked module declaring the named handler.
        attachments : RuntimeAttachments
            Runtime handler bindings for this invocation.

        Raises
        ------
        ValueError
            If the module lacks the graph and word-specific name.
        TypeError
            If the named handler does not have the checked Compute type.
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
        """Return the built-in validator for the exact report carrier.

        Parameters
        ----------
        type_ : TypeExpr
            The carrier requested by the reference machine.

        Returns
        -------
        RuntimeValidator or None
            The canonical product validator for this report.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = ["RegularStringEPRequest", "RegularStringEPRuntimeProvider"]
