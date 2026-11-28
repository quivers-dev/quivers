"""Shared typing and execution for checked sparse relation operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quivers.core.algebras import LOG_PROB, MARKOV, Semiring
from quivers.core.laws import SEMIRING_LAWS, TIMES_COMMUTATIVE
from quivers.core.objects import StringSet
from quivers.core.relations import (
    ComposedRelation,
    RationalRelation,
    auto_intersect,
    closure,
    concatenate,
    condition,
    intersect,
    invert,
    join,
    project,
    trim,
    union,
)
from quivers.core.relations.calculus import _JoinLayout
from quivers.core.relations.topology import _validate_tape_words
from quivers.qiec.relations import RationalRelationSpec

type _UnaryOperation = Literal[
    "project", "condition", "invert", "trim", "closure", "auto_intersect"
]
type _BinaryOperation = Literal["compose", "union", "concatenate", "intersect", "join"]
_UNARY_OPERATIONS = (
    "project",
    "condition",
    "invert",
    "trim",
    "closure",
    "auto_intersect",
)


@dataclass(frozen=True, slots=True)
class _RelationInterface:
    """The typed boundary of a sparse relation.

    Parameters
    ----------
    tapes : tuple of StringSet
        Nominal tape types in order.
    input_tapes : int
        The number of leading input tapes.
    algebra : Semiring
        The trusted scalar algebra for all paths.
    """

    tapes: tuple[StringSet, ...]
    input_tapes: int
    algebra: Semiring

    @classmethod
    def of(cls, source: RationalRelationSpec) -> _RelationInterface:
        """Read a primitive declaration's typed boundary.

        Parameters
        ----------
        source : RationalRelationSpec
            The checked primitive relation.

        Returns
        -------
        _RelationInterface
            The original tape layout and algebra.
        """
        algebra = source._algebra
        assert algebra is not None
        return cls(source.topology.tapes, source.input_tapes, algebra)

    def select(self, order: tuple[int, ...]) -> _RelationInterface:
        """Retain tapes in their relative order.

        Parameters
        ----------
        order : tuple of int
            The increasing positions of the retained tapes.

        Returns
        -------
        _RelationInterface
            The restricted layout with its updated input arity.
        """
        return _RelationInterface(
            tuple(self.tapes[index] for index in order),
            sum(index < self.input_tapes for index in order),
            self.algebra,
        )


def _unary_layout(
    source: _RelationInterface,
    operation: _UnaryOperation,
    positions: tuple[int, ...],
    word: tuple[int, ...] | None,
) -> _RelationInterface:
    """Check a unary recipe and infer its result tape layout.

    Parameters
    ----------
    source : _RelationInterface
        The original typed tape layout.
    operation : str
        The unary sparse operation.
    positions : tuple of int
        Retained, fixed, or equal tape positions, as specified by the operation.
    word : tuple of int or None
        The fixed word of a conditioning operation.

    Returns
    -------
    _RelationInterface
        The transformed tape layout.

    Raises
    ------
    ValueError
        The operation arguments or fixed tape word are invalid.
    TypeError
        The tapes selected for equality have different nominal types.
    """
    count = len(source.tapes)
    if not isinstance(positions, tuple) or any(
        type(index) is not int or not 0 <= index < count for index in positions
    ):
        raise ValueError("transformed relation positions must name valid tapes")
    if operation == "condition":
        if len(positions) != 1 or count < 2 or not isinstance(word, tuple):
            raise ValueError(
                "conditioning requires one fixed tape word and a remaining tape"
            )
        (tape,) = positions
        _validate_tape_words(
            source.tapes, tuple(word if index == tape else () for index in range(count))
        )
        return source.select(tuple(index for index in range(count) if index != tape))
    if word is not None:
        raise ValueError("only conditioning accepts a fixed tape word")
    if operation == "project":
        if not positions or any(b <= a for a, b in zip(positions, positions[1:])):
            raise ValueError("projection requires increasing distinct tape positions")
        return source.select(positions)
    if operation == "auto_intersect":
        if len(positions) != 2 or positions[0] == positions[1]:
            raise ValueError("auto-intersection requires two distinct tape positions")
        first, second = positions
        if source.tapes[first] != source.tapes[second]:
            raise TypeError("auto-intersection requires the same nominal tape type")
        return source
    if positions:
        raise ValueError("this transformed relation operation takes no tape positions")
    if operation == "invert":
        split = source.input_tapes
        return _RelationInterface(
            source.tapes[split:] + source.tapes[:split], count - split, source.algebra
        )
    return source


def _binary_layout(
    left: _RelationInterface,
    right: _RelationInterface,
    operation: _BinaryOperation,
    shared_tapes: tuple[int, ...] | None,
    keep_shared: bool,
) -> _RelationInterface:
    """Check a binary recipe and infer its result tape layout.

    Parameters
    ----------
    left, right : _RelationInterface
        The original typed layouts in operand order.
    operation : str
        The binary sparse operation.
    shared_tapes : tuple of int or None
        Two shared positions for an explicit join.
    keep_shared : bool
        Whether the join retains the shared tape.

    Returns
    -------
    _RelationInterface
        The transformed tape layout.

    Raises
    ------
    TypeError
        The tape or scalar types do not satisfy the operation's laws.
    ValueError
        The operation arguments or composition layouts are invalid.
    """
    if operation != "join" and (shared_tapes is not None or keep_shared):
        raise ValueError("only join accepts shared tape positions and retention")
    if operation == "compose":
        if any(
            len(operand.tapes) != 2 or operand.input_tapes != 1
            for operand in (left, right)
        ):
            raise ValueError(
                "composition requires one input and one output tape per operand"
            )
        if left.tapes[1] != right.tapes[0]:
            raise ValueError(
                "composition operands require the same nominal shared tape"
            )
        if left.algebra is not right.algebra or left.algebra not in (MARKOV, LOG_PROB):
            raise ValueError(
                "composition requires one shared Markov or LogProb algebra"
            )
        return _RelationInterface((left.tapes[0], right.tapes[1]), 1, left.algebra)
    if left.algebra is not right.algebra:
        raise TypeError("binary relation transforms require one shared algebra")
    if operation in ("union", "concatenate"):
        if left.tapes != right.tapes or left.input_tapes != right.input_tapes:
            raise TypeError("relation expressions require identical ordered tape types")
        return left
    if operation == "intersect":
        if (
            len(left.tapes) != 1
            or len(right.tapes) != 1
            or left.input_tapes != right.input_tapes
        ):
            raise TypeError(
                "intersection requires one tape per operand with the same orientation"
            )
        l_tape = r_tape = 0
    else:
        if (
            not isinstance(shared_tapes, tuple)
            or len(shared_tapes) != 2
            or any(
                type(index) is not int or not 0 <= index < len(source.tapes)
                for index, source in zip(shared_tapes, (left, right), strict=True)
            )
        ):
            raise ValueError("join requires one valid tape position per operand")
        l_tape, r_tape = shared_tapes
    if left.tapes[l_tape] != right.tapes[r_tape]:
        raise TypeError("binary products require the same nominal shared tape")
    if not left.algebra.establishes(*SEMIRING_LAWS, TIMES_COMMUTATIVE):
        raise TypeError("binary products require an established commutative semiring")
    layout = _JoinLayout.build(
        left, right, l_tape, r_tape, keep_shared=keep_shared or operation == "intersect"
    )
    sources = left, right
    return _RelationInterface(
        tuple(sources[side].tapes[tape] for side, tape in layout.visible),
        layout.input_tapes,
        left.algebra,
    )


def _apply_unary(
    source: RationalRelation,
    operation: _UnaryOperation,
    positions: tuple[int, ...],
    word: tuple[int, ...] | None,
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Execute a unary operation on a previously checked layout.

    Parameters
    ----------
    source : RationalRelation
        The original sparse relation.
    operation : str
        The unary sparse operation.
    positions : tuple of int
        Retained, fixed, or equal tape positions, as specified by the operation.
    word : tuple of int or None
        The fixed word of a conditioning operation.
    max_states, max_work : int
        Bounds for sparse conditioning or signed-delay tape equality.

    Returns
    -------
    RationalRelation
        The transformed relation with its live original weights.
    """
    match operation:
        case "project":
            return project(source, positions)
        case "condition":
            assert word is not None
            return condition(
                source, positions[0], word, max_states=max_states, max_work=max_work
            )
        case "invert":
            return invert(source)
        case "trim":
            return trim(source)
        case "closure":
            return closure(source)
        case "auto_intersect":
            first, second = positions
            return auto_intersect(
                source, first, second, max_states=max_states, max_work=max_work
            )


def _apply_binary(
    left: RationalRelation,
    right: RationalRelation,
    operation: _BinaryOperation,
    shared_tapes: tuple[int, ...] | None,
    keep_shared: bool,
    *,
    max_states: int,
    max_work: int,
) -> RationalRelation:
    """Execute a binary operation on previously checked layouts.

    Parameters
    ----------
    left, right : RationalRelation
        The ordered operand relations.
    operation : str
        The binary sparse operation.
    shared_tapes : tuple of int or None
        Two shared positions for an explicit join.
    keep_shared : bool
        Whether the join retains the shared tape.
    max_states, max_work : int
        The native sparse product limits.

    Returns
    -------
    RationalRelation
        The transformed relation with live original weights.
    """
    match operation:
        case "compose":
            return ComposedRelation(left, right).materialize(
                max_states=max_states, max_work=max_work
            )
        case "union":
            return union(left, right)
        case "concatenate":
            return concatenate(left, right)
        case "intersect":
            return intersect(left, right, max_states=max_states, max_work=max_work)
        case "join":
            assert shared_tapes is not None
            return join(
                left,
                right,
                left_tape=shared_tapes[0],
                right_tape=shared_tapes[1],
                keep_shared=keep_shared,
                max_states=max_states,
                max_work=max_work,
            )
