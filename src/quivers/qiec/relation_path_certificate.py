"""Checked native reports for numerical Markov path weight certificates."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import cached_property
from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import MARKOV, DivergentClosureError, UndecidedClosureError
from quivers.core.relations import (
    PathWeightCertificate,
    ProductiveEpsilonCycleError,
    RelationAccuracyError,
    RelationLimitError,
)
from quivers.core.relations.accuracy import (
    _UnsupportedAccuracyPrecision,
    _markov_accuracy_plan,
)
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.relations import FixedRelationRequest, RelationWeightBindings
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexLiteral,
    TypeExpr,
    product_type,
)

type _Ratio = tuple[int, int]
type _EncodedRatio = tuple[str, str]
type _Witness = tuple[_EncodedRatio, _EncodedRatio, _EncodedRatio, _EncodedRatio]
type _Report = tuple[
    str, str, torch.Tensor, torch.Tensor, _Witness, torch.Tensor, torch.Tensor
]
_MISSING = ("", "")


def _encode(ratio: Fraction | None) -> _EncodedRatio:
    """Encode a whole rational bound without narrowing its integers.

    Parameters
    ----------
    ratio : Fraction or None
        The exact bound, or None if this coordinate is unavailable.

    Returns
    -------
    tuple of str
        The hexadecimal numerator and denominator, or two empty strings.
    """
    return _MISSING if ratio is None else (hex(ratio.numerator), hex(ratio.denominator))


def _unavailable(status: str, error: Exception) -> _Report:
    """Represent a checked refusal without a scalar or rational witness.

    Parameters
    ----------
    status : str
        The report's unavailable status.
    error : Exception
        The specific failed condition supplied by the core evaluator.

    Returns
    -------
    tuple
        The failed status, diagnostic, and unavailable report fields.
    """
    score = torch.zeros((), dtype=torch.float64)
    absent = score.new_tensor(False, dtype=torch.bool)
    return (
        status,
        str(error),
        absent,
        score,
        (_MISSING, _MISSING, _MISSING, _MISSING),
        absent,
        score.new_zeros(3, dtype=torch.int64),
    )


class FixedMarkovPathCertificateRequest(QiecModel):
    """Request an exact forward-error witness for an evaluated Markov pair score.

    Parameters
    ----------
    observation : FixedRelationRequest
        The original nominal Markov pair request and its live weight identities.
    max_exact_bits, max_exact_work : int
        Exact arithmetic limits for the rational witness.
    max_closure_cells : int
        Aggregate local star cell limit, including auxiliary states.
    absolute_tolerance, relative_tolerance : tuple of int or None
        Exact numerator and positive denominator of each required tolerance.
        At least one tolerance is required.
    """

    observation: FixedRelationRequest = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_exact_bits: int
    max_exact_work: int
    max_closure_cells: int
    absolute_tolerance: _Ratio | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    relative_tolerance: _Ratio | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_certificate_request(self) -> None:
        """Check the declared value objective and representable evidence limits.

        Raises
        ------
        ValueError
            The observation, tolerances, or resource limits are unsupported.
        """
        if (
            self.observation.operation != "pair_weight"
            or self.observation.relation.algebra != MARKOV.subject
        ):
            raise ValueError("path certificates require a checked Markov pair weight")
        maximum = torch.iinfo(torch.int64).max
        for limit in (self.max_exact_bits, self.max_exact_work, self.max_closure_cells):
            if type(limit) is not int or not 0 <= limit <= maximum:
                raise ValueError("path certificate resource limits must fit in Int")
        if self.absolute_tolerance is None and self.relative_tolerance is None:
            raise ValueError(
                "path certification requires an absolute or relative tolerance"
            )
        for ratio in (self.absolute_tolerance, self.relative_tolerance):
            if ratio is None:
                continue
            if (
                not isinstance(ratio, tuple)
                or len(ratio) != 2
                or any(type(value) is not int for value in ratio)
                or ratio[0] < 0
                or ratio[1] <= 0
            ):
                raise ValueError(
                    "path certificate tolerances require nonnegative rational pairs"
                )
            if max(ratio[0].bit_length(), ratio[1].bit_length()) > self.max_exact_bits:
                raise ValueError("a path certificate tolerance exceeds max_exact_bits")

    @cached_property
    def _tolerances(self) -> tuple[Fraction | None, Fraction | None]:
        """Read the exact declared tolerances without floating-point conversion.

        Returns
        -------
        tuple of Fraction or None
            Absolute and relative requirements in that order.
        """
        absolute, relative = self.absolute_tolerance, self.relative_tolerance
        return (
            None if absolute is None else Fraction(*absolute),
            None if relative is None else Fraction(*relative),
        )

    @property
    def handler_name(self) -> str:
        """Derive the checked nominal owner for the observation and requirements.

        Returns
        -------
        str
            The foreign handler name for the exact topology, words, and tolerances.
        """
        from quivers.qiec.serialization import dumps

        identity = (
            self.observation.relation,
            self.observation.words,
            tuple(_encode(ratio) for ratio in self._tolerances),
        )
        return (
            f"__markov_path_certificate_{sha256(dumps(identity).encode()).hexdigest()}"
        )

    @property
    def result_type(self) -> TypeExpr:
        """Give the value and exact-rational evidence carrier.

        Returns
        -------
        TypeExpr
            Status, reason, availability, score, four encoded rational
            bounds, relative-bound availability, and three work counters.
        """
        rows = IndexLiteral(value=4, sort=NAT)
        pair = IndexLiteral(value=2, sort=NAT)
        witness = tensor_type(STRING, (rows, pair))
        work = tensor_type(INT, (IndexLiteral(value=3, sort=NAT),))
        return product_type(STRING, STRING, BOOL, REAL, witness, BOOL, work)


class FixedMarkovPathCertificateRuntimeProvider:
    """Return a value-only report through checked native Compute.

    Available reports contain exact signed hexadecimal numerator and
    positive denominator strings for the interval endpoints and error
    bounds. A relative bound is present only when requested and verified.
    Unavailable reports identify the failed condition and have no bounds.

    Parameters
    ----------
    request : FixedMarkovPathCertificateRequest
        The checked observation, tolerances, and certification limits.
    parameters : RelationWeightBindings
        The original live owners, snapshotted independently on each call.
    """

    name = "fixed-markov-path-certificate"

    def __init__(
        self,
        request: FixedMarkovPathCertificateRequest,
        parameters: RelationWeightBindings,
    ) -> None:
        self._request = request
        self._relation = request.observation.relation.bind(parameters)
        self._handler_name = request.handler_name
        self._result_type = request.result_type

    @cached_property
    def _plan(self) -> Callable[[], PathWeightCertificate]:
        """Compile the reusable graph and exact arithmetic policy.

        Returns
        -------
        Callable
            A live scalar evaluator with its exact forward-error witness.
        """
        request = self._request
        observation = request.observation
        absolute, relative = request._tolerances
        return _markov_accuracy_plan(
            self._relation,
            observation.words,
            max_states=observation.max_states,
            max_work=observation.max_work,
            max_closure_size=observation.max_closure_size,
            max_exact_bits=request.max_exact_bits,
            max_exact_work=request.max_exact_work,
            max_closure_cells=request.max_closure_cells,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
        )

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Evaluate one certificate request against the current live weights.

        Parameters
        ----------
        argument : object
            The Unit argument supplied to Compute.
        _context : ClauseContext
            The checked machine's clause context.

        Returns
        -------
        tuple
            The typed value and exact evidence or an explicit unavailable report.

        Raises
        ------
        TypeError
            If the supplied argument is outside Unit.
        """
        if argument is not None:
            raise TypeError("a path weight certificate takes a unit argument")
        try:
            with torch.no_grad():
                proof = self._plan()
        except RelationLimitError as error:
            return _unavailable("resource_limit", error)
        except DivergentClosureError as error:
            return _unavailable("divergent", error)
        except UndecidedClosureError as error:
            return _unavailable("undecided_closure", error)
        except ProductiveEpsilonCycleError as error:
            return _unavailable("unsupported_closure", error)
        except _UnsupportedAccuracyPrecision as error:
            return _unavailable("unsupported_precision", error)
        except RelationAccuracyError as error:
            return _unavailable("unverified_accuracy", error)
        score = proof.weight.detach()
        witness = (
            _encode(proof.lower),
            _encode(proof.upper),
            _encode(proof.absolute_error),
            _encode(proof.relative_error),
        )
        return (
            "certified",
            "",
            score.new_tensor(True, dtype=torch.bool),
            score,
            witness,
            score.new_tensor(proof.relative_error is not None, dtype=torch.bool),
            score.new_tensor(
                (proof.exact_work, proof.exact_bits, proof.closure_cells),
                dtype=torch.int64,
            ),
        )

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach to the exact ordinary authored foreign Compute handler.

        Parameters
        ----------
        module : QiecModule
            The checked module declaring the nominal certificate handler.
        attachments : RuntimeAttachments
            The runtime binding table for this invocation.

        Raises
        ------
        ValueError
            The module lacks the correct certificate handler name.
        TypeError
            The matching declaration has an incompatible Compute contract.
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
        """Supply the canonical validator for the exact report product.

        Parameters
        ----------
        type_ : TypeExpr
            The carrier requested by the reference machine.

        Returns
        -------
        RuntimeValidator or None
            The checked report validator when the carrier matches.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


__all__ = [
    "FixedMarkovPathCertificateRequest",
    "FixedMarkovPathCertificateRuntimeProvider",
]
