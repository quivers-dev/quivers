"""Checked native value certificates for fixed-word LogProb observations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import cached_property
from hashlib import sha256
from typing import TYPE_CHECKING

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
from quivers.core.relations.accuracy import _UnsupportedAccuracyPrecision
from quivers.core.relations.log_accuracy import (
    _logprob_accuracy_plan,
    _validate_decimal_settings,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_path_certificate import _MISSING, _encode, _EncodedRatio
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import FixedRelationRequest, RelationWeightBindings
from quivers.qiec.types import (
    BOOL,
    INT,
    STRING,
    IndexLiteral,
    TypeExpr,
    product_type,
)

if TYPE_CHECKING:
    from quivers.qiec.centered_string_certificate import (
        CenteredCyclicStringLogProbCertificateRequest,
        CenteredStringLogProbCertificateRequest,
    )


type _Witness = tuple[_EncodedRatio, _EncodedRatio, _EncodedRatio]
type _Report = tuple[
    str, str, torch.Tensor, torch.Tensor, _Witness, torch.Tensor, torch.Tensor
]


class FixedLogProbPathCertificateRequest(QiecModel):
    """Request absolute log accuracy for a checked native pair weight.

    Parameters
    ----------
    observation : FixedRelationRequest
        The acyclic LogProb pair observation and its original weight owners.
    absolute_log_tolerance : tuple of int
        Nonnegative numerator and positive denominator in log units.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Significant digits, symmetric adjusted exponent bound, and number
        of Decimal calls and binary64 conversions allowed.
    max_exact_bits, max_exact_work : int
        Integer bit and operation bounds for the rational report.
    """

    observation: FixedRelationRequest = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    absolute_log_tolerance: tuple[int, int] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    decimal_precision: int
    max_decimal_exponent: int
    max_decimal_work: int
    max_exact_bits: int
    max_exact_work: int

    @dx.model_validator()
    def _validate_certificate_request(self) -> None:
        """Check the value objective, tolerance, and resource policy.

        Raises
        ------
        ValueError
            The observation, tolerance, or resource policy is invalid.
        """
        observation = self.observation
        if (
            observation.operation != "pair_weight"
            or observation.relation.algebra != LOG_PROB.subject
            or observation.max_closure_size is not None
        ):
            raise ValueError(
                "LogProb path certificates require an acyclic checked LogProb pair weight"
            )
        _validate_numeric_policy(self)

    @property
    def handler_name(self) -> str:
        """Derive the handler identity for the exact numerical objective.

        Returns
        -------
        str
            The name binding the relation, ordered words, and canonical tolerance.
        """
        return _certificate_handler(self, "__logprob_path_certificate_")

    @property
    def result_type(self) -> TypeExpr:
        """Give the checked score and exact-rational evidence carrier.

        Returns
        -------
        TypeExpr
            Status, diagnostic, score availability, log score, three
            rational rows, endpoint availability, and three work counters.
        """
        return _certificate_result_type()


class FixedCyclicLogProbPathCertificateRequest(QiecModel):
    """Request log accuracy with explicitly bounded productive epsilon closure.

    Parameters
    ----------
    observation : FixedRelationRequest
        The checked LogProb pair observation, original weight owners,
        and required local closure order limit.
    absolute_log_tolerance : tuple of int
        Nonnegative numerator and positive denominator in log units.
    max_closure_cells : int
        The aggregate local star cell bound, including auxiliary states.
    decimal_precision, max_decimal_exponent, max_decimal_work : int
        Significant digits, symmetric adjusted exponent bound, and number
        of primitive calls and exact Decimal input conversions permitted.
    max_exact_bits, max_exact_work : int
        Integer bit and operation bounds for the rational report.
    """

    observation: FixedRelationRequest = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    absolute_log_tolerance: tuple[int, int] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    max_closure_cells: int
    decimal_precision: int
    max_decimal_exponent: int
    max_decimal_work: int
    max_exact_bits: int
    max_exact_work: int

    @dx.model_validator()
    def _validate_certificate_request(self) -> None:
        """Check the value objective, tolerance, and closure policy.

        Raises
        ------
        ValueError
            The observation, tolerance, or resource policy is invalid.
        """
        observation = self.observation
        if (
            observation.operation != "pair_weight"
            or observation.relation.algebra != LOG_PROB.subject
            or observation.max_closure_size is None
        ):
            raise ValueError(
                "cyclic LogProb path certificates require a checked LogProb pair "
                "weight with a closure order limit"
            )
        if (
            type(self.max_closure_cells) is not int
            or not 0 <= self.max_closure_cells <= torch.iinfo(torch.int64).max
        ):
            raise ValueError("max_closure_cells must be a nonnegative Int")
        _validate_numeric_policy(self)

    @property
    def handler_name(self) -> str:
        """Derive the identity of the exact bounded cyclic objective.

        Returns
        -------
        str
            The handler binding the relation, ordered words, and tolerance.
        """
        return _certificate_handler(self, "__cyclic_logprob_path_certificate_")

    @property
    def result_type(self) -> TypeExpr:
        """Give the checked log score and exact-rational evidence carrier.

        Returns
        -------
        TypeExpr
            Status, diagnostic, score availability, log score, three
            rational rows, endpoint availability, and three work counters.
        """
        return _certificate_result_type()


type _CertificateRequest = (
    FixedLogProbPathCertificateRequest | FixedCyclicLogProbPathCertificateRequest
)


def _validate_numeric_policy(
    request: _CertificateRequest
    | CenteredStringLogProbCertificateRequest
    | CenteredCyclicStringLogProbCertificateRequest,
) -> None:
    """Check the shared arithmetic policy.

    Parameters
    ----------
    request : _CertificateRequest or CenteredStringLogProbCertificateRequest or CenteredCyclicStringLogProbCertificateRequest
        A LogProb request with declared tolerance and arithmetic limits.

    Raises
    ------
    ValueError
        A tolerance or resource declaration is unsupported.
    """
    _validate_decimal_settings(request.decimal_precision, request.max_decimal_exponent)
    for limit in (
        request.max_decimal_work,
        request.max_exact_bits,
        request.max_exact_work,
    ):
        if type(limit) is not int or not 0 <= limit <= torch.iinfo(torch.int64).max:
            raise ValueError("LogProb path certificate resource limits must fit in Int")
    ratio = request.absolute_log_tolerance
    if (
        not isinstance(ratio, tuple)
        or len(ratio) != 2
        or any(type(value) is not int for value in ratio)
        or ratio[0] < 0
        or ratio[1] <= 0
    ):
        raise ValueError("absolute_log_tolerance requires a nonnegative rational pair")
    if max(ratio[0].bit_length(), ratio[1].bit_length()) > request.max_exact_bits:
        raise ValueError("absolute_log_tolerance exceeds max_exact_bits")


def _certificate_handler(request: _CertificateRequest, prefix: str) -> str:
    """Bind a value-certificate handler to the exact requested objective.

    Parameters
    ----------
    request : FixedLogProbPathCertificateRequest or FixedCyclicLogProbPathCertificateRequest
        The declaration of relation identity, ordered words, and tolerance.
    prefix : str
        The handler namespace for the request's closure policy.

    Returns
    -------
    str
        A stable handler name that a compiled source module may request.
    """
    from quivers.qiec.serialization import dumps

    identity = (
        request.observation.relation,
        request.observation.words,
        _encode(Fraction(*request.absolute_log_tolerance)),
    )
    return f"{prefix}{sha256(dumps(identity).encode()).hexdigest()}"


def _certificate_result_type() -> TypeExpr:
    """Give the common typed value-certificate report.

    Returns
    -------
    TypeExpr
        The product of status, score, exact witness, and work fields.
    """
    three = IndexLiteral(value=3, sort=NAT)
    pair = IndexLiteral(value=2, sort=NAT)
    return product_type(
        STRING,
        STRING,
        BOOL,
        LOG_WEIGHT,
        tensor_type(STRING, (three, pair)),
        BOOL,
        tensor_type(INT, (three,)),
    )


class _LogProbCertificateRuntime:
    """Share checked attachment and live snapshot evaluation across closure modes.

    Parameters
    ----------
    request : FixedLogProbPathCertificateRequest or FixedCyclicLogProbPathCertificateRequest
        The validated observation, arithmetic policy, and nominal identity.
    parameters : RelationWeightBindings
        The original scalar weight owners.
    """

    def __init__(
        self,
        request: _CertificateRequest,
        parameters: RelationWeightBindings,
    ) -> None:
        self._request = request
        self._relation = request.observation.relation.bind(parameters)
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    @cached_property
    def _plan(self) -> Callable[[], LogPathWeightCertificate]:
        """Compile the structural plan without retaining any weight values.

        Returns
        -------
        Callable
            An evaluator using one fresh live snapshot per call.
        """
        request = self._request
        observation = request.observation
        return _logprob_accuracy_plan(
            self._relation,
            observation.words,
            max_states=observation.max_states,
            max_work=observation.max_work,
            absolute_log_tolerance=Fraction(*request.absolute_log_tolerance),
            decimal_precision=request.decimal_precision,
            max_decimal_exponent=request.max_decimal_exponent,
            max_decimal_work=request.max_decimal_work,
            max_exact_bits=request.max_exact_bits,
            max_exact_work=request.max_exact_work,
            max_closure_size=observation.max_closure_size,
            max_closure_cells=(
                request.max_closure_cells
                if isinstance(request, FixedCyclicLogProbPathCertificateRequest)
                else None
            ),
        )

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Evaluate the result and its certificate through native Compute.

        Parameters
        ----------
        argument : object
            The Unit argument.
        _context : ClauseContext
            The checked machine's clause context.

        Returns
        -------
        tuple
            A detached score and certificate or an unavailable result.

        Raises
        ------
        TypeError
            The Compute argument is outside Unit.
        """
        if argument is not None:
            raise TypeError("a LogProb path certificate takes a unit argument")
        try:
            with torch.no_grad():
                proof = self._plan()
        except RelationLimitError as error:
            return _unavailable("resource_limit", error)
        except DivergentClosureError as error:
            return _unavailable("divergent_closure", error)
        except UndecidedClosureError as error:
            return _unavailable("undecided_closure", error)
        except ProductiveEpsilonCycleError as error:
            return _unavailable("unsupported_closure", error)
        except _UnsupportedAccuracyPrecision as error:
            return _unavailable("unsupported_precision", error)
        except RelationAccuracyError as error:
            return _unavailable("unverified_accuracy", error)
        return _pack_log_certificate(proof)

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach to the exact ordinary checked foreign Compute handler.

        Parameters
        ----------
        module : QiecModule
            The module declaring this exact value-certificate handler.
        attachments : RuntimeAttachments
            The machine's runtime binding table.

        Raises
        ------
        ValueError
            The module lacks the handler for the requested nominal objective.
        TypeError
            The handler has an incompatible Compute contract.
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
        """Supply the validator for the exact native report product.

        Parameters
        ----------
        type_ : TypeExpr
            The queried runtime carrier.

        Returns
        -------
        RuntimeValidator or None
            The canonical product validator when the carrier matches.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


class FixedLogProbPathCertificateRuntimeProvider(_LogProbCertificateRuntime):
    """Bind one checked acyclic LogProb value report to original live owners.

    The provider reuses the structural observation plan. Each evaluation
    uses one binary64 weight snapshot to score and to verify the result.
    The native report is detached and supplies value-only assurance.

    Parameters
    ----------
    request : FixedLogProbPathCertificateRequest
        The checked observation, tolerance, and numerical limits.
    parameters : RelationWeightBindings
        The binding to the original scalar owners.
    """

    name = "fixed-logprob-path-certificate"

    def __init__(
        self,
        request: FixedLogProbPathCertificateRequest,
        parameters: RelationWeightBindings,
    ) -> None:
        if not isinstance(request, FixedLogProbPathCertificateRequest):
            raise TypeError("the acyclic provider requires an acyclic request")
        super().__init__(request, parameters)


class FixedCyclicLogProbPathCertificateRuntimeProvider(_LogProbCertificateRuntime):
    """Bind one checked cyclic LogProb value report to original live owners.

    Each evaluation scores and verifies the same evaluated binary64
    vector. A proven divergent local star has a different status from
    an upper enclosure that cannot establish convergence. The native
    report is detached and supplies value-only assurance.

    Parameters
    ----------
    request : FixedCyclicLogProbPathCertificateRequest
        The exact checked observation, arithmetic policy, and closure limits.
    parameters : RelationWeightBindings
        The binding to the original scalar owners.
    """

    name = "fixed-cyclic-logprob-path-certificate"

    def __init__(
        self,
        request: FixedCyclicLogProbPathCertificateRequest,
        parameters: RelationWeightBindings,
    ) -> None:
        if not isinstance(request, FixedCyclicLogProbPathCertificateRequest):
            raise TypeError("the cyclic provider requires a bounded cyclic request")
        super().__init__(request, parameters)


def _pack_log_certificate(proof: LogPathWeightCertificate) -> _Report:
    """Pack the finite witness and score, with a distinct log-zero result.

    Parameters
    ----------
    proof : LogPathWeightCertificate
        The delivered value and its exact rational bounds.

    Returns
    -------
    tuple
        The typed, value-only LogProb certificate report.
    """
    score = proof.weight.detach()
    return (
        "certified",
        "",
        score.new_tensor(True, dtype=torch.bool),
        score,
        (
            _encode(proof.lower),
            _encode(proof.upper),
            _encode(proof.absolute_log_error),
        ),
        score.new_tensor(proof.lower is not None, dtype=torch.bool),
        score.new_tensor(
            (proof.decimal_work, proof.exact_work, proof.exact_bits),
            dtype=torch.int64,
        ),
    )


def _unavailable(status: str, error: Exception) -> _Report:
    """Represent an unavailable score and exact witness.

    Parameters
    ----------
    status : str
        The category of the refusal.
    error : Exception
        The specific failed condition from the core evaluator.

    Returns
    -------
    tuple
        The refusal and unavailable report fields.
    """
    score = torch.zeros((), dtype=torch.float64)
    absent = score.new_tensor(False, dtype=torch.bool)
    return (
        status,
        str(error),
        absent,
        score,
        (_MISSING, _MISSING, _MISSING),
        absent,
        score.new_zeros(3, dtype=torch.int64),
    )


__all__ = [
    "FixedCyclicLogProbPathCertificateRequest",
    "FixedCyclicLogProbPathCertificateRuntimeProvider",
    "FixedLogProbPathCertificateRequest",
    "FixedLogProbPathCertificateRuntimeProvider",
]
