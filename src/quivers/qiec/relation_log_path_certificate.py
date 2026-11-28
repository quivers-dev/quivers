"""Checked native value certificates for acyclic LogProb observations."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import cached_property
from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
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
        _validate_decimal_settings(self.decimal_precision, self.max_decimal_exponent)
        for limit in (self.max_decimal_work, self.max_exact_bits, self.max_exact_work):
            if type(limit) is not int or not 0 <= limit <= torch.iinfo(torch.int64).max:
                raise ValueError(
                    "LogProb path certificate resource limits must fit in Int"
                )
        ratio = self.absolute_log_tolerance
        if (
            not isinstance(ratio, tuple)
            or len(ratio) != 2
            or any(type(value) is not int for value in ratio)
            or ratio[0] < 0
            or ratio[1] <= 0
        ):
            raise ValueError(
                "absolute_log_tolerance requires a nonnegative rational pair"
            )
        if max(ratio[0].bit_length(), ratio[1].bit_length()) > self.max_exact_bits:
            raise ValueError("absolute_log_tolerance exceeds max_exact_bits")

    @cached_property
    def _tolerance(self) -> Fraction:
        """Read the exact canonical tolerance.

        Returns
        -------
        Fraction
            The required absolute error in log units.
        """
        return Fraction(*self.absolute_log_tolerance)

    @property
    def handler_name(self) -> str:
        """Derive the handler identity for the exact numerical objective.

        Returns
        -------
        str
            The name binding the relation, ordered words, and canonical tolerance.
        """
        from quivers.qiec.serialization import dumps

        identity = (
            self.observation.relation,
            self.observation.words,
            _encode(self._tolerance),
        )
        return (
            f"__logprob_path_certificate_{sha256(dumps(identity).encode()).hexdigest()}"
        )

    @property
    def result_type(self) -> TypeExpr:
        """Give the checked score and exact-rational evidence carrier.

        Returns
        -------
        TypeExpr
            Status, diagnostic, score availability, log score, three
            rational rows, endpoint availability, and three work counters.
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


class FixedLogProbPathCertificateRuntimeProvider:
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
            absolute_log_tolerance=request._tolerance,
            decimal_precision=request.decimal_precision,
            max_decimal_exponent=request.max_decimal_exponent,
            max_decimal_work=request.max_decimal_work,
            max_exact_bits=request.max_exact_bits,
            max_exact_work=request.max_exact_work,
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
        except ProductiveEpsilonCycleError as error:
            return _unavailable("unsupported_closure", error)
        except _UnsupportedAccuracyPrecision as error:
            return _unavailable("unsupported_precision", error)
        except RelationAccuracyError as error:
            return _unavailable("unverified_accuracy", error)
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
    "FixedLogProbPathCertificateRequest",
    "FixedLogProbPathCertificateRuntimeProvider",
]
