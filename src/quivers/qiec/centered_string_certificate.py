"""Checked value-only certificates for centered acyclic string ratios."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import cached_property
from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, DivergentClosureError, UndecidedClosureError
from quivers.core.relations import (
    LogPathWeightCertificate,
    ProductiveEpsilonCycleError,
    RelationAccuracyError,
    RelationLimitError,
)
from quivers.core.relations._centered import _UndefinedCenteredNormalizer
from quivers.core.relations._centered_accuracy import _centered_accuracy_plan
from quivers.core.relations.accuracy import _UnsupportedAccuracyPrecision
from quivers.core.relations.centered_cyclic import _centered_cyclic_accuracy_plan
from quivers.core.relations.log_accuracy import _UnverifiedUpperStar
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.centered_string_runtime import CenteredStringLogProbRequest
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_log_path_certificate import (
    _certificate_result_type,
    _pack_log_certificate,
    _Report,
    _unavailable,
    _validate_numeric_policy,
)
from quivers.qiec.relation_path_certificate import _encode
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import FixedRelationRequest, RelationWeightBindings
from quivers.qiec.types import TypeExpr


class CenteredStringLogProbCertificateRequest(QiecModel):
    """Request a value-only certificate for a centered string log ratio.

    Parameters
    ----------
    observation : CenteredStringLogProbRequest
        The checked conditional or tree word score and its original owners.
    absolute_log_tolerance : tuple of int
        Nonnegative numerator and positive denominator in log units.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Precision, exponent bound, and combined work for directed residuals.
    max_exact_bits, max_exact_work : int
        Integer bit and operation caps for the exact verification pass.
    """

    observation: CenteredStringLogProbRequest = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    absolute_log_tolerance: tuple[int, int] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    decimal_precision: int
    max_decimal_exponent: int
    max_decimal_work: int
    max_exact_bits: int
    max_exact_work: int

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the nested checked objective and arithmetic limits.

        Raises
        ------
        TypeError
            The observation does not identify a checked centered ratio.
        ValueError
            A numerical tolerance or work limit is invalid.
        """
        if not isinstance(self.observation, CenteredStringLogProbRequest):
            raise TypeError(
                "a centered ratio certificate requires a centered observation"
            )
        _validate_numeric_policy(self)

    @property
    def handler_name(self) -> str:
        """Identify the checked observation and its exact absolute tolerance.

        Returns
        -------
        str
            The distinct foreign handler name for this value guarantee.
        """
        from quivers.qiec.serialization import dumps

        identity = (
            self.observation.handler_name,
            _encode(Fraction(*self.absolute_log_tolerance)),
        )
        digest = sha256(dumps(identity).encode("utf-8")).hexdigest()
        return f"__centered_string_logprob_certificate_{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Give the checked carrier of a score and exact rational endpoints.

        Returns
        -------
        TypeExpr
            Status, diagnostic, score availability, log score, three
            rational rows, endpoint availability, and three work counters.
        """
        return _certificate_result_type()


class CenteredCyclicStringLogProbCertificateRequest(QiecModel):
    """Request a bounded checked fixed-input cyclic likelihood certificate.

    Parameters
    ----------
    observation : FixedRelationRequest
        The original LogProb conditional objective, words, and local star order limit.
    max_center_occurrences, max_dyadic_work : int
        Aggregate centered observation occurrence and graph visit caps.
    max_closure_cells : int
        Combined numeric closure cells, including auxiliary states.
    absolute_log_tolerance : tuple of int
        Nonnegative numerator and positive denominator in log units.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Directed precision, exponent limit, and combined Decimal work.
    max_exact_bits, max_exact_work : int
        Integer bit and operation caps for gauges, endpoint stars, and the value test.
    """

    observation: FixedRelationRequest = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_center_occurrences: int
    max_dyadic_work: int
    max_closure_cells: int
    absolute_log_tolerance: tuple[int, int] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    decimal_precision: int
    max_decimal_exponent: int
    max_decimal_work: int
    max_exact_bits: int
    max_exact_work: int

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the original conditional objective and bounded verification policy.

        Raises
        ------
        ValueError
            The objective, closure, or arithmetic declaration is outside the supported domain.
        """
        observation = self.observation
        if (
            not isinstance(observation, FixedRelationRequest)
            or observation.operation != "conditional_log_prob"
            or observation.relation.algebra != LOG_PROB.subject
            or observation.max_closure_size is None
        ):
            raise ValueError(
                "a centered cyclic certificate requires a LogProb conditional request with a local star bound"
            )
        for name, limit in (
            ("max_center_occurrences", self.max_center_occurrences),
            ("max_dyadic_work", self.max_dyadic_work),
            ("max_closure_cells", self.max_closure_cells),
        ):
            if type(limit) is not int or not 0 <= limit <= torch.iinfo(torch.int64).max:
                raise ValueError(f"{name} must be a nonnegative Int")
        _validate_numeric_policy(self)

    @property
    def handler_name(self) -> str:
        """Identify the original conditional objective and rational tolerance.

        Returns
        -------
        str
            The distinct checked handler name for the cyclic value guarantee.
        """
        from quivers.qiec.serialization import dumps

        identity = (
            self.observation.relation,
            self.observation.words,
            _encode(Fraction(*self.absolute_log_tolerance)),
        )
        digest = sha256(dumps(identity).encode("utf-8")).hexdigest()
        return f"__centered_cyclic_string_logprob_certificate_{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Give the checked carrier of a score and its exact rational endpoints.

        Returns
        -------
        TypeExpr
            Status, diagnostic, score availability, log score, three rational rows,
            endpoint availability, and three arithmetic work counters.
        """
        return _certificate_result_type()


class CenteredStringLogProbCertificateRuntimeProvider:
    """Bind a centered value certificate to live original weight owners.

    Parameters
    ----------
    request : CenteredStringLogProbCertificateRequest or CenteredCyclicStringLogProbCertificateRequest
        The checked ratio, tolerance and directed arithmetic policy.
    parameters : RelationWeightBindings
        The stable live scalar owner mapping for the relation or graph.
    """

    name = "centered-string-logprob-certificate"

    def __init__(
        self,
        request: CenteredStringLogProbCertificateRequest
        | CenteredCyclicStringLogProbCertificateRequest,
        parameters: RelationWeightBindings,
    ) -> None:
        self._request = request
        self._parameters = dict(parameters)
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    @cached_property
    def _plan(self) -> Callable[[], LogPathWeightCertificate]:
        """Compile the reusable observables without reading their values.

        Returns
        -------
        Callable
            A centered verifier reading one fresh weight snapshot per call.
        """
        request = self._request
        tolerance = Fraction(*request.absolute_log_tolerance)
        numeric = {
            "decimal_precision": request.decimal_precision,
            "max_decimal_exponent": request.max_decimal_exponent,
            "max_decimal_work": request.max_decimal_work,
            "max_exact_bits": request.max_exact_bits,
            "max_exact_work": request.max_exact_work,
        }
        if isinstance(request, CenteredCyclicStringLogProbCertificateRequest):
            observation = request.observation
            order = observation.max_closure_size
            assert order is not None
            relation = observation.relation.bind(self._parameters)
            split = relation.topology.input_tapes
            return _centered_cyclic_accuracy_plan(
                relation,
                observation.words[:split],
                observation.words[split:],
                absolute_log_tolerance=tolerance,
                max_states=observation.max_states,
                max_work=observation.max_work,
                max_center_occurrences=request.max_center_occurrences,
                max_dyadic_work=request.max_dyadic_work,
                max_closure_size=order,
                max_closure_cells=request.max_closure_cells,
                **numeric,
            )
        return _centered_accuracy_plan(
            request.observation._binding(self._parameters),
            absolute_log_tolerance=tolerance,
            **numeric,
        )

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Evaluate the value or return its specific unavailable outcome.

        Parameters
        ----------
        argument : object
            The Unit argument of the fixed ratio query.
        _context : ClauseContext
            The checked Compute clause context.

        Returns
        -------
        tuple
            The delivered score and witness, or a precise refusal.

        Raises
        ------
        TypeError
            The argument does not inhabit Unit.
        """
        if argument is not None:
            raise TypeError("a centered string certificate takes a unit argument")
        try:
            with torch.no_grad():
                proof = self._plan()
        except RelationLimitError as error:
            return _unavailable("resource_limit", error)
        except ProductiveEpsilonCycleError as error:
            return _unavailable("unsupported_closure", error)
        except _UnsupportedAccuracyPrecision as error:
            return _unavailable("unsupported_precision", error)
        except _UndefinedCenteredNormalizer as error:
            return _unavailable("undefined_normalizer", error)
        except OverflowError as error:
            return _unavailable("numerical_range", error)
        except DivergentClosureError as error:
            return _unavailable("divergent_closure", error)
        except UndecidedClosureError as error:
            return _unavailable("undecided_closure", error)
        except _UnverifiedUpperStar as error:
            return _unavailable("unverified_closure", error)
        except RelationAccuracyError as error:
            return _unavailable("unverified_accuracy", error)
        return _pack_log_certificate(proof)

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Install the runtime on the exact checked foreign computation.

        Parameters
        ----------
        module : QiecModule
            The module declaring this observation and value guarantee.
        attachments : RuntimeAttachments
            The invocation's checked attachment table.

        Raises
        ------
        TypeError
            The expected declaration does not have the checked contract.
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
        """Supply the validator for this request's exact result type.

        Parameters
        ----------
        type_ : TypeExpr
            The requested runtime result type.

        Returns
        -------
        RuntimeValidator or None
            The canonical validator when the type matches this report.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = [
    "CenteredCyclicStringLogProbCertificateRequest",
    "CenteredStringLogProbCertificateRequest",
    "CenteredStringLogProbCertificateRuntimeProvider",
]
