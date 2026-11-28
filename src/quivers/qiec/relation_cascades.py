"""Checked acyclic source recipes for cascaded sparse string operations."""

from __future__ import annotations

from functools import cached_property
from typing import Literal, cast

import didactic.api as dx
import torch
from torch import nn

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.laws import ClaimSubject
from quivers.core.relations import (
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    _weight_snapshot,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec._relation_operations import (
    _UNARY_OPERATIONS,
    _BinaryOperation,
    _RelationInterface,
    _UnaryOperation,
    _apply_binary,
    _apply_unary,
    _binary_layout,
    _unary_layout,
)
from quivers.qiec.relations import (
    RationalRelationSpec,
    RelationWeightBindings,
    _weight_bindings,
)


class RelationCascadeStep(QiecModel):
    """One named primitive relation or an ordered operation in a cascade.

    Parameters
    ----------
    name : str
        The nominal source declaration name.
    primitive : RationalRelationSpec or None
        The original topology and weight owners of a primitive step.
    operation : str or None
        The sparse operation of a derived step.
    operands : tuple of int
        The earlier step indices, in operation order.
    positions : tuple of int
        Tape positions for project, condition, and join.
    word : tuple of int or None
        The fixed tape word for conditioning.
    keep_shared : bool
        Whether a join retains its synchronized tape.
    """

    name: str
    primitive: RationalRelationSpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    operation: (
        Literal[
            "project",
            "condition",
            "invert",
            "trim",
            "closure",
            "compose",
            "union",
            "concatenate",
            "intersect",
            "join",
        ]
        | None
    ) = None
    operands: tuple[int, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    positions: tuple[int, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    word: tuple[int, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    keep_shared: bool = False

    @dx.model_validator()
    def _validate_step(self) -> None:
        """Check local operation arity and the disjoint primitive/derived forms.

        Raises
        ------
        TypeError
            The primitive value is not a checked relation.
        ValueError
            The declaration name, operand indices, or arguments are invalid.
        """
        if not self.name:
            raise ValueError("a relation cascade step requires a name")
        if not isinstance(self.operands, tuple) or any(
            type(index) is not int or index < 0 for index in self.operands
        ):
            raise ValueError("cascade operands require nonnegative integer indices")
        if not isinstance(self.positions, tuple):
            raise ValueError("cascade tape positions require an ordered tuple")
        if self.operation is None:
            if not isinstance(self.primitive, RationalRelationSpec):
                raise TypeError("a primitive cascade step requires a checked relation")
            if (
                self.operands
                or self.positions
                or self.word is not None
                or self.keep_shared
            ):
                raise ValueError(
                    "a primitive cascade step takes no operation arguments"
                )
            return
        if self.primitive is not None:
            raise ValueError("a derived cascade step takes ordered operands")
        unary = self.operation in _UNARY_OPERATIONS
        if len(self.operands) != (1 if unary else 2):
            raise ValueError("a cascade operation has the wrong operand arity")
        if self.operation != "condition" and self.word is not None:
            raise ValueError("only conditioning accepts a fixed tape word")
        if self.operation != "join" and self.keep_shared:
            raise ValueError("only join can retain a shared tape")

    def _interface(self, inputs: list[_RelationInterface]) -> _RelationInterface:
        """Infer this step's nominal layout.

        Parameters
        ----------
        inputs : list of _RelationInterface
            The already checked earlier step layouts.

        Returns
        -------
        _RelationInterface
            The ordered tape types, orientation, and scalar semiring.
        """
        if self.primitive is not None:
            return _RelationInterface.of(self.primitive)
        assert self.operation is not None
        left = inputs[self.operands[0]]
        if len(self.operands) == 1:
            return _unary_layout(
                left, cast(_UnaryOperation, self.operation), self.positions, self.word
            )
        return _binary_layout(
            left,
            inputs[self.operands[1]],
            cast(_BinaryOperation, self.operation),
            self.positions if self.operation == "join" else self.positions or None,
            self.keep_shared,
        )


def _records(topology: RelationTopology) -> int:
    """Count tape, arc, per-tape label, boundary, and slot records.

    Parameters
    ----------
    topology : RelationTopology
        The sparse relation's immutable topology.

    Returns
    -------
    int
        The number of stored records in the work contract.
    """
    return (
        len(topology.tapes)
        + len(topology.arcs) * (1 + len(topology.tapes))
        + len(topology.initial)
        + len(topology.final)
        + len(topology.weight_names)
    )


def _check_size(states: int, records: int, max_states: int, max_work: int) -> None:
    """Enforce one topology's state and record limits.

    Parameters
    ----------
    states, records : int
        The actual or predicted state and record counts.
    max_states, max_work : int
        The corresponding state and record limits.

    Raises
    ------
    RelationLimitError
        A count exceeds one of its limits.
    """
    if states > max_states:
        raise RelationLimitError("cascade topology exceeds max_states")
    if records > max_work:
        raise RelationLimitError("cascade topology exceeds max_work records")


def _preflight_expanding_step(
    operation: str,
    left: RelationTopology,
    right: RelationTopology | None,
    max_states: int,
    max_work: int,
) -> None:
    """Bound the exact output size of expanding linear operations.

    Parameters
    ----------
    operation : str
        The selected sparse operation.
    left, right : RelationTopology or None
        The already bounded ordered input topologies.
    max_states, max_work : int
        The output topology's state and record limits.
    """
    # These operations allocate output topology directly rather than walking
    # a native resource-limited product. Bound their exact output size first.
    if operation == "closure":
        width = 1 + len(left.tapes)
        count = len(left.initial) + len(left.final)
        records = _records(left) - count + count * width + 3
        _check_size(left.n_states + 1, records, max_states, max_work)
    elif operation in ("union", "concatenate"):
        assert right is not None
        states = left.n_states + right.n_states
        records = _records(left) + _records(right) - len(left.tapes)
        if operation == "concatenate":
            states += 1
            marks = len(left.final) + len(right.initial)
            records += marks * len(left.tapes)
        _check_size(states, records, max_states, max_work)


class _CascadeWeights(nn.Module):
    """Read a constructed cascade in dependency order.

    Parameters
    ----------
    relations : tuple of RationalRelation
        The bound sparse relations in checked step order.
    """

    def __init__(self, relations: tuple[RationalRelation, ...]) -> None:
        super().__init__()
        self._relations = relations
        self.sources = nn.ModuleList(dict.fromkeys(item.module() for item in relations))

    def forward(self) -> torch.Tensor:
        """Evaluate fresh original weights in a call-local snapshot.

        Returns
        -------
        torch.Tensor
            A differentiable weight vector for the result topology.
        """
        # Visit parents after children so every operand read hits a live tensor
        # in this evaluation's cache, including very deep and shared recipes.
        with _weight_snapshot():
            for relation in self._relations[:-1]:
                relation._weights()
            return self._relations[-1]._weights()


class RelationCascadeSpec(QiecModel):
    """An acyclic checked sparse relation program with the last step as its result.

    Every step is used by the result. The step table preserves names,
    dependency sharing, ordered operand occurrences, and primitive
    parameter ownership. Each native operation uses the same sparse
    compiler as a single checked relation transformation.

    Parameters
    ----------
    steps : tuple of RelationCascadeStep
        Dependency-ordered steps, each referring only to earlier indices.
    """

    steps: tuple[RelationCascadeStep, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_dag(self) -> None:
        """Reject malformed, cyclic, unreachable or ill-typed steps.

        Raises
        ------
        ValueError
            A step reference, name, or checked operation is invalid.
        """
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("a cascade requires a nonempty ordered step table")
        names = set()
        used: set[int] = set()
        for index, step in enumerate(self.steps):
            if not isinstance(step, RelationCascadeStep):
                raise TypeError("a cascade requires checked step records")
            if step.name in names:
                raise ValueError(f"duplicate relation cascade step {step.name!r}")
            names.add(step.name)
            if any(operand >= index for operand in step.operands):
                raise ValueError(
                    f"cascade step {step.name!r} must reference only earlier steps"
                )
            used.update(step.operands)
        if len(used) != len(self.steps) - 1:
            raise ValueError("the relation cascade retains an unused step")
        _ = self._layout

    @cached_property
    def _layout(self) -> _RelationInterface:
        """Type-check the ordered relation DAG once.

        Returns
        -------
        _RelationInterface
            The result layout and scalar semiring.

        Raises
        ------
        ValueError
            A step violates the operation's tape or algebra contract.
        """
        inputs: list[_RelationInterface] = []
        for step in self.steps:
            try:
                inputs.append(step._interface(inputs))
            except (TypeError, ValueError) as error:
                raise ValueError(f"cascade step {step.name!r}: {error}") from error
        return inputs[-1]

    @property
    def algebra(self) -> ClaimSubject:
        """Return the trusted identity of the cascade's scalar semiring.

        Returns
        -------
        ClaimSubject
            The algebra of every primitive and operation in the result.
        """
        return self._layout.algebra.subject

    def _check_budgets(self, max_states: int, max_work: int) -> None:
        """Refuse oversized declarations before binding.

        Parameters
        ----------
        max_states : int
            Maximum states in an intermediate topology.
        max_work : int
            Maximum primitive topology records and distinct steps.

        Raises
        ------
        RelationLimitError
            A primitive or step count exceeds the budget.
        """
        _validate_limits(max_states, max_work)
        if len(self.steps) > max_work:
            raise RelationLimitError("cascade step count exceeds max_work")
        for step in self.steps:
            if step.primitive is not None:
                top = step.primitive.topology
                _check_size(top.n_states, _records(top), max_states, max_work)

    def bind(
        self,
        parameters: RelationWeightBindings,
        *,
        max_states: int,
        max_work: int,
    ) -> RationalRelation:
        """Build one bounded sparse topology per step with live original weights.

        Construction uses at most one visit per shared declaration. The
        per-step bound limits native construction states, probes or
        matched arcs, and topology records. The same work limit bounds
        the number of distinct steps. Evaluation visits this table
        once and reads each weight source once per call, caching only
        within that differentiable call. The final fixed observation
        has its own limits.

        Parameters
        ----------
        parameters : RelationWeightBindings
            The live scalar tensors for original primitive parameter IDs.
        max_states : int
            The state limit for each sparse intermediate.
        max_work : int
            The operation work and intermediate topology record limit,
            and the number of distinct steps.

        Returns
        -------
        RationalRelation
            The cascade result with live reads of its primitive weights.

        Raises
        ------
        RelationLimitError
            Construction, storage, or the step count exceeds the budget.
        ValueError
            Different IDs alias a tensor or bindings disagree on tensor domain.
        """
        self._check_budgets(max_states, max_work)
        primitives = tuple(
            step.primitive for step in self.steps if step.primitive is not None
        )
        _weight_bindings(
            (identity for spec in primitives for identity in spec.weight_parameters),
            parameters,
            description="cascade",
        )
        primitive_machines = iter(tuple(spec.bind(parameters) for spec in primitives))
        results: list[RationalRelation] = []
        for step in self.steps:
            if step.primitive is not None:
                results.append(next(primitive_machines))
                continue
            assert step.operation is not None
            left = results[step.operands[0]]
            right = results[step.operands[1]] if len(step.operands) == 2 else None
            _preflight_expanding_step(
                step.operation,
                left.topology,
                None if right is None else right.topology,
                max_states,
                max_work,
            )
            if right is None:
                result = _apply_unary(
                    left,
                    cast(_UnaryOperation, step.operation),
                    step.positions,
                    step.word,
                    max_states=max_states,
                    max_work=max_work,
                )
            else:
                result = _apply_binary(
                    left,
                    right,
                    cast(_BinaryOperation, step.operation),
                    step.positions or None,
                    step.keep_shared,
                    max_states=max_states,
                    max_work=max_work,
                )
            top = result.topology
            _check_size(top.n_states, _records(top), max_states, max_work)
            results.append(result)
        root = results[-1]
        return RationalRelation(
            root.topology, _CascadeWeights(tuple(results)), algebra=root.algebra
        )


__all__ = ["RelationCascadeSpec", "RelationCascadeStep"]
