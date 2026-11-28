"""Checked native exact MAP with paired local subproblems."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.relations import PairedAlignmentMapResult, RelationLimitError
from quivers.core.relations.alignment_groups import StringAlignmentTrace
from quivers.qiec._models import QiecModel
from quivers.qiec.alignment_map_runtime import (
    _INT_BYTES,
    _INT_MAX,
    _map_report_fields,
    _MapReportRuntime,
    _Report,
    _Witness,
    _report_work,
    _validate_assignment_shape,
)
from quivers.qiec.canonical import tensor_type
from quivers.qiec.contracts import ParameterId
from quivers.qiec.evaluator import ClauseContext
from quivers.qiec.kinds import NAT
from quivers.qiec.relation_path_certificate import _MISSING, _encode, _EncodedRatio
from quivers.qiec.string_graphs import StringFactorGraphSpec
from quivers.qiec.string_solvers import JointAlignmentMapDecomposition
from quivers.qiec.types import INT, STRING, IndexLiteral, TypeExpr, product_type

type _ProductShapes = tuple[tuple[int, int] | None, ...]
type _PairedReport = tuple[
    str,
    str,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    _Witness,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[_EncodedRatio, ...],
    torch.Tensor,
]


class PairedAlignmentMapRequest(QiecModel):
    """Request exact native MAP with declared pair joins and original traces.

    Parameters
    ----------
    graph : StringFactorGraphSpec
        The authored original graph and live nominal factor owners.
    solver : JointAlignmentMapDecomposition
        The ordered singleton and pair partition, solver policy, and budgets.
    word_capacity : int
        Allocated token positions per variable's word.
    max_assignment_cells : int
        Combined output cell limit for words and lengths.
    max_ratio_bits : int
        Combined numerator and denominator bit limit for the common evidence.
    arc_capacity : int
        Allocated original occurrence positions per factor trace.
    max_trace_cells : int
        Combined output cell limit for integer trace and product tensors.
    max_trace_ratio_bits : int
        Combined numerator and denominator bit limit for original trace scores.
    """

    graph: StringFactorGraphSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    solver: JointAlignmentMapDecomposition = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    word_capacity: int
    max_assignment_cells: int
    max_ratio_bits: int
    arc_capacity: int
    max_trace_cells: int
    max_trace_ratio_bits: int

    @dx.model_validator()
    def _validate_output(self) -> None:
        """Check the static word, trace and group allocation extents.

        Raises
        ------
        ValueError
            The declared capacities or the fixed output allocation are invalid.
        """
        limits = (
            self.word_capacity,
            self.max_assignment_cells,
            self.max_ratio_bits,
            self.arc_capacity,
            self.max_trace_cells,
            self.max_trace_ratio_bits,
        )
        if any(
            type(limit) is not int or not 0 <= limit <= _INT_MAX for limit in limits
        ):
            raise ValueError("paired MAP report limits must be nonnegative Int values")
        _validate_assignment_shape(
            len(self.graph.variables), self.word_capacity, self.max_assignment_cells
        )
        cells = len(self.graph.factors) * (self.arc_capacity + 3)
        cells += 2 * len(self.solver.groups)
        if cells > self.max_trace_cells:
            raise ValueError("paired MAP report exceeds max_trace_cells")
        if cells > _INT_MAX // _INT_BYTES:
            raise ValueError("paired MAP trace storage exceeds the Int range")

    @property
    def objective(self) -> Literal["joint_alignment_map"]:
        """Give the objective of the exact paired solver.

        Returns
        -------
        Literal["joint_alignment_map"]
            The joint maximum over original factor traces and words.
        """
        return self.solver.objective

    @property
    def handler_name(self) -> str:
        """Derive the exact paired request's foreign handler identity.

        Returns
        -------
        str
            A canonical name binding the graph, ordered decomposition and layout.
        """
        from quivers.qiec.serialization import dumps

        identity = self.graph, self.solver, self.word_capacity, self.arc_capacity
        return "__paired_alignment_map_" + sha256(dumps(identity).encode()).hexdigest()

    @property
    def result_type(self) -> TypeExpr:
        """Give the common report fields followed by exact paired evidence.

        Returns
        -------
        TypeExpr
            The verified search report, original boundaries and lengths,
            padded original arcs, rational trace scores, and group sizes.
        """
        factors = IndexLiteral(value=len(self.graph.factors), sort=NAT)
        groups = IndexLiteral(value=len(self.solver.groups), sort=NAT)
        capacity = IndexLiteral(value=self.arc_capacity, sort=NAT)
        pair = IndexLiteral(value=2, sort=NAT)
        three = IndexLiteral(value=3, sort=NAT)
        return product_type(
            *_map_report_fields(len(self.graph.variables), self.word_capacity),
            tensor_type(INT, (factors, three)),
            tensor_type(INT, (factors, capacity)),
            tensor_type(STRING, (factors, pair)),
            tensor_type(INT, (groups, pair)),
        )


class PairedAlignmentMapRuntimeProvider(_MapReportRuntime):
    """Attach the paired exact local solver and its original factor witnesses.

    Parameters
    ----------
    request : PairedAlignmentMapRequest
        The exact graph, pair partition, solver and encoding policy.
    parameters : Mapping of ParameterId to torch.Tensor
        Original live weights supplying the graph on each call.
    """

    name = "paired-alignment-map"

    def __init__(
        self,
        request: PairedAlignmentMapRequest,
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

    def _evaluate(self, argument: object, _context: ClauseContext) -> _PairedReport:
        """Run and report the verified paired search and primitive traces.

        Parameters
        ----------
        argument : object
            The Unit argument to Compute.
        _context : ClauseContext
            The checked machine context.

        Returns
        -------
        tuple
            The checked report and original witnesses, or resource refusal.

        Raises
        ------
        TypeError
            The argument is outside Unit.
        """
        if argument is not None:
            raise TypeError("a paired alignment MAP report takes a unit argument")
        request = self._request
        try:
            with torch.no_grad():
                result = request.solver.solve(request.graph, self._parameters)
        except RelationLimitError as error:
            return self._pack(self._unavailable("resource_limit", str(error)))

        if reason := self._trace_limit(result):
            work = _report_work(result.search)
            if any(
                type(value) is not int or not 0 <= value <= _INT_MAX for value in work
            ):
                return self._pack(
                    self._unavailable(
                        "report_limit", "alignment MAP work counts exceed Int"
                    )
                )
            return self._pack(self._unavailable("report_limit", reason, work))
        common = self._report(result.search)
        if common[0] == "report_limit":
            return self._pack(common)
        return self._pack(common, result.factor_traces or (), result.product_shapes)

    def _trace_limit(self, result: PairedAlignmentMapResult) -> str | None:
        """Check scope consistency and encoding limits before packing the evidence.

        Parameters
        ----------
        result : PairedAlignmentMapResult
            The result of the portable paired solver and its exact verifier.

        Returns
        -------
        str or None
            A diagnostic when a complete encoding would exceed its bounds.

        Raises
        ------
        ArithmeticError
            The original trace order, assignment or product metadata is inconsistent.
        """
        request = self._request
        groups = request.solver.groups
        shapes = result.product_shapes
        if (
            result.groups != groups
            or len(shapes) != len(groups)
            or any(
                (shape is None) != (len(group) == 1)
                for group, shape in zip(groups, shapes, strict=True)
            )
        ):
            raise ArithmeticError(
                "paired MAP product shapes differ from the requested groups"
            )
        if any(
            type(value) is not int or not 0 <= value <= _INT_MAX
            for shape in shapes
            if shape is not None
            for value in shape
        ):
            return "paired MAP product sizes exceed Int"
        traces = result.factor_traces
        assignment = result.search.assignment
        if traces is None or assignment is None:
            if (traces is None) != (assignment is None):
                raise ArithmeticError(
                    "paired MAP assignment and trace availability disagree"
                )
            return None
        words = dict(assignment)
        if (
            len(traces) != len(request.graph.factors)
            or len(words) != len(self._variables)
            or words.keys() != set(self._variables)
            or any(
                trace.words != tuple(words[name] for name in scope)
                for trace, (_, scope) in zip(traces, request.graph.factors, strict=True)
            )
            or sum(trace.score for trace in traces) != result.search.score
        ):
            raise ArithmeticError(
                "paired MAP original traces differ from the assignment or score"
            )
        if any(len(trace.arc_ids) > request.arc_capacity for trace in traces):
            return "paired MAP trace exceeds arc_capacity"
        if any(
            type(value) is not int or not 0 <= value <= _INT_MAX
            for trace in traces
            for value in (trace.start, trace.stop, *trace.arc_ids)
        ):
            return "paired MAP original occurrence or boundary exceeds Int"
        bits = sum(
            trace.score.numerator.bit_length() + trace.score.denominator.bit_length()
            for trace in traces
        )
        if bits > request.max_trace_ratio_bits:
            return "paired MAP exact scores exceed max_trace_ratio_bits"
        return None

    def _pack(
        self,
        common: _Report,
        traces: tuple[StringAlignmentTrace, ...] = (),
        shapes: _ProductShapes = (),
    ) -> _PairedReport:
        """Allocate the checked paired output after completing its preflight.

        Parameters
        ----------
        common : tuple
            The accepted common search report or an unavailable result.
        traces : tuple
            Validated original factor traces in factor order, when available.
        shapes : tuple
            Pair state and arc sizes in group order when a solve returned.

        Returns
        -------
        tuple
            Common report, original trace metadata, arc rows, scores and sizes.
        """
        request = self._request
        factors = len(request.graph.factors)
        metadata = torch.full((factors, 3), -1, dtype=torch.int64)
        arcs = metadata.new_full((factors, request.arc_capacity), -1)
        scores: list[_EncodedRatio] = [_MISSING] * factors
        for index, trace in enumerate(traces):
            length = len(trace.arc_ids)
            metadata[index] = metadata.new_tensor((trace.start, trace.stop, length))
            arcs[index, :length] = arcs.new_tensor(trace.arc_ids)
            scores[index] = _encode(trace.score)
        products = metadata.new_full((len(request.solver.groups), 2), -1)
        for index, shape in enumerate(shapes):
            if shape is not None:
                products[index] = products.new_tensor(shape)
        return (*common, metadata, arcs, tuple(scores), products)


__all__ = ["PairedAlignmentMapRequest", "PairedAlignmentMapRuntimeProvider"]
