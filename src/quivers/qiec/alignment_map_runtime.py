"""Checked native reports for exact joint alignment MAP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from hashlib import sha256
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.relations import RelationLimitError, StringAlignmentMapResult
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import tensor_type
from quivers.qiec.contracts import ParameterId
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_path_certificate import _MISSING, _encode, _EncodedRatio
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.string_graphs import StringFactorGraphSpec
from quivers.qiec.string_solvers import (
    JointAlignmentMapConfig,
    JointAlignmentMapGapConfig,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    STRING,
    IndexLiteral,
    TypeExpr,
    product_type,
)

type _Witness = tuple[_EncodedRatio, _EncodedRatio, _EncodedRatio, _EncodedRatio]
type _Work = tuple[int, int, int, int, int]
type _Report = tuple[
    str,
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    _Witness,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

_INT_MAX = torch.iinfo(torch.int64).max
_INT_BYTES = torch.iinfo(torch.int64).bits // 8


class JointAlignmentMapRequest(QiecModel):
    """Request one exact-scope joint alignment MAP report from authored QVR.

    The graph and already portable singleton or gap solver remain separate
    fields. The named handler binds both together with the word capacity.
    The exact score and bound refer to the live evaluated graph.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The original nominal tapes, observations, factor occurrences and owners.
    solver : JointAlignmentMapConfig or JointAlignmentMapGapConfig
        The exact local decoder policy and optional objective-gap targets.
    word_capacity : int
        Allocated token positions for each variable's returned word.
    max_assignment_cells : int
        Combined allocation limit for the padded words and their length vector.
    max_ratio_bits : int
        Combined numerator and denominator bit limit for four rational fields.
        The search has separately declared graph and iteration limits.
    """

    graph: StringFactorGraphSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    solver: JointAlignmentMapConfig | JointAlignmentMapGapConfig = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    word_capacity: int
    max_assignment_cells: int
    max_ratio_bits: int

    @dx.model_validator()
    def _validate_report_limits(self) -> None:
        """Preflight the fixed output shape before creating the request.

        Raises
        ------
        ValueError
            The dimensions or declared allocation limits cannot admit the shape.
        """
        limits = (self.word_capacity, self.max_assignment_cells, self.max_ratio_bits)
        if any(
            type(limit) is not int or not 0 <= limit <= _INT_MAX for limit in limits
        ):
            raise ValueError(
                "alignment MAP report limits must be nonnegative Int values"
            )
        _validate_assignment_shape(
            len(self.graph.variables), self.word_capacity, self.max_assignment_cells
        )

    @property
    def objective(self) -> Literal["joint_alignment_map"]:
        """Give the exact objective chosen by the checked request.

        Returns
        -------
        Literal["joint_alignment_map"]
            Joint maximization over one original alignment per factor and its words.
        """
        return self.solver.objective

    @property
    def handler_name(self) -> str:
        """Bind the handler to the exact graph, solver, and word capacity.

        Returns
        -------
        str
            The canonical name used in the source Compute declaration.
        """
        from quivers.qiec.serialization import dumps

        declared = self.graph, self.solver, self.word_capacity
        return "__joint_alignment_map_" + sha256(dumps(declared).encode()).hexdigest()

    @property
    def result_type(self) -> TypeExpr:
        """Give the checked representation of the exact report.

        Returns
        -------
        TypeExpr
            Status, diagnostic, assignment presence, padded symbols, lengths,
            rational rows, agreement and gap flags, and five work counts.
        """
        return product_type(
            *_map_report_fields(len(self.graph.variables), self.word_capacity)
        )


def _validate_assignment_shape(
    variable_count: int, word_capacity: int, max_assignment_cells: int
) -> None:
    """Check the fixed native assignment and length tensor extents.

    Parameters
    ----------
    variable_count : int
        The number of declared variables.
    word_capacity : int
        The number of token positions per row.
    max_assignment_cells : int
        The combined limit on symbol rows and length entries.

    Raises
    ------
    ValueError
        The combined allocation exceeds the declared or runtime range.
    """
    cells = variable_count * (word_capacity + 1)
    if cells > max_assignment_cells:
        raise ValueError("alignment MAP report exceeds max_assignment_cells")
    if cells > _INT_MAX // _INT_BYTES:
        raise ValueError("alignment MAP assignment storage exceeds the Int range")


def _map_report_fields(variable_count: int, word_capacity: int) -> tuple[TypeExpr, ...]:
    """Construct the ordered fields of the common checked MAP report.

    Parameters
    ----------
    variable_count : int
        The number of declared variables.
    word_capacity : int
        The number of allocated token positions per variable.

    Returns
    -------
    tuple of TypeExpr
        The exact status, assignment, rational evidence and work carrier.
    """
    variables = IndexLiteral(value=variable_count, sort=NAT)
    capacity = IndexLiteral(value=word_capacity, sort=NAT)
    four = IndexLiteral(value=4, sort=NAT)
    pair = IndexLiteral(value=2, sort=NAT)
    five = IndexLiteral(value=5, sort=NAT)
    return (
        STRING,
        STRING,
        BOOL,
        tensor_type(INT, (variables, capacity)),
        tensor_type(INT, (variables,)),
        tensor_type(STRING, (four, pair)),
        BOOL,
        BOOL,
        tensor_type(INT, (five,)),
    )


def _report_work(result: StringAlignmentMapResult) -> _Work:
    """Extract the five measured common MAP counters.

    Parameters
    ----------
    result : StringAlignmentMapResult
        The exact verified search result.

    Returns
    -------
    tuple of int
        Iterations, local states and work, and verification states and work.
    """
    return (
        result.iterations,
        result.max_local_states,
        result.local_arc_examinations,
        result.max_verification_states,
        result.verification_arc_examinations,
    )


class _MapReportRuntime(ABC):
    """Share verified search encoding and attachment across exact MAP reports.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The checked factor model and variable declaration order.
    word_capacity, max_ratio_bits : int
        Preflighted assignment shape and the declared ratio encoding limit.
    handler_name : str
        The exact nominal foreign declaration.
    result_type : TypeExpr
        The native report's checked result carrier.
    """

    def __init__(
        self,
        graph: StringFactorGraphSpec,
        word_capacity: int,
        max_ratio_bits: int,
        handler_name: str,
        result_type: TypeExpr,
    ) -> None:
        self._variables = tuple(name for name, _, _, _ in graph.variables)
        self._word_capacity = word_capacity
        self._max_ratio_bits = max_ratio_bits
        self._handler_name = handler_name
        self._result_type = result_type

    @abstractmethod
    def _evaluate(self, argument: object, _context: ClauseContext) -> object:
        """Evaluate one checked native report.

        Parameters
        ----------
        argument : object
            The Unit argument.
        _context : ClauseContext
            The native machine context.

        Returns
        -------
        object
            A value in this report's exact checked carrier.
        """

    def _report(self, result: StringAlignmentMapResult) -> _Report:
        """Preflight and pack a verified solve result without narrowing its scores.

        Parameters
        ----------
        result : StringAlignmentMapResult
            The existing exact search report after verification.

        Returns
        -------
        tuple
            Lossless evidence for representable results, or a report-limit outcome.

        Raises
        ------
        ArithmeticError
            The solver's status, assignment, or verified evidence is inconsistent.
        """
        work = _report_work(result)
        if any(type(value) is not int or not 0 <= value <= _INT_MAX for value in work):
            return self._unavailable(
                "report_limit", "alignment MAP work counts exceed Int"
            )
        assigned = result.assignment is not None
        certified = result.certificate is not None
        if (
            (result.status == "optimal") != certified
            or assigned != (result.score is not None)
            or (certified and (not assigned or result.score != result.upper_bound))
            or (result.status == "gap" and not result.gap_target_met)
        ):
            raise ArithmeticError("alignment MAP returned inconsistent evidence")
        words: tuple[tuple[int, ...], ...] = ()
        if result.assignment is not None:
            by_name = dict(result.assignment)
            if by_name.keys() != set(self._variables) or len(by_name) != len(
                result.assignment
            ):
                raise ArithmeticError(
                    "alignment MAP assignment differs from graph variables"
                )
            words = tuple(by_name[name] for name in self._variables)
            if any(len(word) > self._word_capacity for word in words):
                return self._unavailable(
                    "report_limit", "alignment MAP word exceeds word_capacity", work
                )
            if any(
                type(token) is not int or not 0 <= token <= _INT_MAX
                for word in words
                for token in word
            ):
                return self._unavailable(
                    "report_limit", "alignment MAP token exceeds Int", work
                )
        lower, upper = result.score, result.upper_bound
        absolute, relative = result.absolute_gap, result.relative_gap
        bits = sum(
            value.numerator.bit_length() + value.denominator.bit_length()
            for value in (lower, upper, absolute, relative)
            if value is not None
        )
        if bits > self._max_ratio_bits:
            return self._unavailable(
                "report_limit", "alignment MAP evidence exceeds max_ratio_bits", work
            )
        padded, lengths = self._assignment(words)
        return (
            result.status,
            "",
            padded.new_tensor(assigned, dtype=torch.bool),
            padded,
            lengths,
            (_encode(lower), _encode(upper), _encode(absolute), _encode(relative)),
            padded.new_tensor(certified, dtype=torch.bool),
            padded.new_tensor(result.gap_target_met, dtype=torch.bool),
            padded.new_tensor(work, dtype=torch.int64),
        )

    def _assignment(
        self, words: tuple[tuple[int, ...], ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill the already admitted assignment buffers.

        Parameters
        ----------
        words : tuple
            Validated words in graph declaration order. An empty outer tuple
            stands for an absent assignment.

        Returns
        -------
        tuple of torch.Tensor
            Padded integer rows and exact lengths. Absent lengths are -1.
        """
        count = len(self._variables)
        padded = torch.full((count, self._word_capacity), -1, dtype=torch.int64)
        lengths = padded.new_full((count,), -1)
        for index, word in enumerate(words):
            lengths[index] = len(word)
            padded[index, : len(word)] = padded.new_tensor(word)
        return padded, lengths

    def _unavailable(
        self, status: str, reason: str, work: _Work = (0, 0, 0, 0, 0)
    ) -> _Report:
        """Encode a resource or representation refusal with absent evidence.

        Parameters
        ----------
        status, reason : str
            The category and the diagnostic.
        work : tuple of int
            Available measured counts from a completed solver invocation.

        Returns
        -------
        tuple
            The fixed-shape report with no assignment, score, or certificate.
        """
        padded, lengths = self._assignment(())
        absent = padded.new_tensor(False, dtype=torch.bool)
        return (
            status,
            reason,
            absent,
            padded,
            lengths,
            (_MISSING, _MISSING, _MISSING, _MISSING),
            absent,
            absent,
            padded.new_tensor(work, dtype=torch.int64),
        )

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Bind only the exact declared alignment MAP Compute.

        Parameters
        ----------
        module : QiecModule
            The compiled source containing the nominal request handler.
        attachments : RuntimeAttachments
            The current invocation's native handler table.

        Raises
        ------
        ValueError
            The exact handler name is absent or ambiguous.
        TypeError
            The declaration has an incompatible Compute boundary.
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
        """Supply the exact report's canonical product validator.

        Parameters
        ----------
        type_ : TypeExpr
            The checked runtime carrier to validate.

        Returns
        -------
        RuntimeValidator or None
            The product validator when the requested carrier matches.
        """
        if type_ != self._result_type:
            return None
        return CoreRuntimeProvider().validator_for(type_)


class JointAlignmentMapRuntimeProvider(_MapReportRuntime):
    """Run the declared exact joint MAP solver through ordinary native QVR.

    Each invocation uses the original live parameter identities. The
    portable solver verifies its witnesses before the report is encoded.
    The report contains discrete words and exact rational evidence.

    Parameters
    ----------
    request : JointAlignmentMapRequest
        The checked authored graph, solver and report bounds.
    parameters : Mapping of ParameterId to torch.Tensor
        Original live LogProb scalar weights used by the graph.
    """

    name = "joint-alignment-map"

    def __init__(
        self,
        request: JointAlignmentMapRequest,
        parameters: Mapping[ParameterId, torch.Tensor],
    ) -> None:
        super().__init__(
            request.graph,
            request.word_capacity,
            request.max_ratio_bits,
            request.handler_name,
            request.result_type,
        )
        self._request = request
        self._parameters = parameters

    def _evaluate(self, argument: object, _context: ClauseContext) -> _Report:
        """Run the same exact solver and independently checked bound as the host.

        Parameters
        ----------
        argument : object
            The Unit argument to Compute.
        _context : ClauseContext
            The active clause context.

        Returns
        -------
        tuple
            The typed report or a resource refusal.

        Raises
        ------
        TypeError
            The argument is outside Unit.
        """
        if argument is not None:
            raise TypeError("an alignment MAP report takes a unit argument")
        request = self._request
        try:
            with torch.no_grad():
                result = request.solver.solve(request.graph, self._parameters)
        except RelationLimitError as error:
            return self._unavailable("resource_limit", str(error))
        return self._report(result)


__all__ = ["JointAlignmentMapRequest", "JointAlignmentMapRuntimeProvider"]
