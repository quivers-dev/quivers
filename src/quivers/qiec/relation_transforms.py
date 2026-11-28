"""Checked recipes for unary operations on sparse rational relations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.laws import SEMIRING_LAWS, TIMES_COMMUTATIVE, ClaimSubject
from quivers.core.relations import (
    FixedStringQuery,
    RationalRelation,
    Words,
    closure,
    concatenate,
    condition,
    intersect,
    join,
    invert,
    project,
    trim,
    union,
)
from quivers.core.relations.calculus import _JoinLayout
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec, _bind_relation_operands
from quivers.qiec.types import REAL, TypeExpr


class UnaryRelationTransformSpec(QiecModel):
    """A checked unary recipe that retains the source's original weight identities.

    Projection sums over erased tapes. Conditioning keeps paths that
    match one fixed tape. Inversion exchanges input and output tape
    order; trimming removes structurally unproductive records.
    Closure adds all finite repetitions of the original paths.

    Parameters
    ----------
    source : RationalRelationSpec
        The checked original topology and live parameter identities.
    operation : str
        The specific supported topology transformation.
    positions : tuple of int
        For projection, the increasing retained tape positions. For
        conditioning, the single fixed tape. Other operations omit it.
    word : tuple of int or None
        The fixed word for conditioning, including an empty tuple.
    """

    source: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operation: Literal["project", "condition", "invert", "trim", "closure"]
    positions: tuple[int, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    word: tuple[int, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_operation(self) -> None:
        """Check operation parameters against the underlying tape types.

        Raises
        ------
        ValueError
            The recipe, tape positions, or conditioning word are invalid.
        """
        if not isinstance(self.source, RationalRelationSpec):
            raise TypeError(
                "a transformed relation requires a checked source specification"
            )
        n_tapes = len(self.source.tapes)
        if not isinstance(self.positions, tuple) or any(
            type(i) is not int or not 0 <= i < n_tapes for i in self.positions
        ):
            raise ValueError("transformed relation positions must name valid tapes")
        if self.operation == "condition":
            if (
                len(self.positions) != 1
                or n_tapes < 2
                or not isinstance(self.word, tuple)
            ):
                raise ValueError(
                    "conditioning requires one fixed tape word and a remaining tape"
                )
            self._validate_words(((),) * (n_tapes - 1))
        elif self.word is not None:
            raise ValueError("only conditioning accepts a fixed tape word")
        elif self.operation == "project":
            positions = self.positions
            if not positions or any(b <= a for a, b in zip(positions, positions[1:])):
                raise ValueError(
                    "projection requires increasing distinct tape positions"
                )
        elif self.positions:
            raise ValueError(
                "this transformed relation operation takes no tape positions"
            )

    def _tape_order(self) -> tuple[int, ...]:
        """Return the source tape positions in the transformed order.

        Returns
        -------
        tuple of int
            One source position per transformed tape.
        """
        count = len(self.source.tapes)
        split = self.source.input_tapes
        if self.operation == "project":
            return self.positions
        if self.operation == "condition":
            return tuple(i for i in range(count) if i != self.positions[0])
        if self.operation == "invert":
            return (*range(split, count), *range(split))
        return tuple(range(count))

    def _validate_words(self, words: Words) -> None:
        """Check transformed words through the source's tape validator.

        Parameters
        ----------
        words : Words
            The observed word on each transformed tape, in order.

        Raises
        ------
        ValueError
            The observed words have the wrong arity or include an invalid symbol.
        """
        positions = self._tape_order()
        if len(positions) != len(words):
            raise ValueError(
                "a transformed query needs exactly one word per retained tape"
            )
        padded: list[tuple[int, ...]] = [()] * len(self.source.tapes)
        for position, word in zip(positions, words, strict=True):
            padded[position] = word
        if self.word is not None:
            padded[self.positions[0]] = self.word
        self.source.topology._validate_words(tuple(padded))

    def bind(
        self,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        max_states: int,
        max_work: int,
    ) -> RationalRelation:
        """Construct and bind the transformed sparse relation.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original scalar weight bindings.
        max_states : int
            State limit for a conditioned topology.
        max_work : int
            Arc examination limit for a conditioned topology.

        Returns
        -------
        RationalRelation
            The transformed topology with the original weights.
        """
        _validate_limits(max_states, max_work)
        source = self.source.bind(parameters)
        match self.operation:
            case "project":
                return project(source, self.positions)
            case "condition":
                assert self.word is not None
                return condition(
                    source,
                    self.positions[0],
                    self.word,
                    max_states=max_states,
                    max_work=max_work,
                )
            case "invert":
                return invert(source)
            case "trim":
                return trim(source)
            case "closure":
                return closure(source)


class BinaryRelationTransformSpec(QiecModel):
    """A checked two-operand recipe with ordered typed weight sources.

    Union adds masses with distinct path occurrences. Concatenation
    sums over word splittings. Intersection and join sum over
    compatible pairs, using the bounded epsilon-filtered product.

    Parameters
    ----------
    left, right : RationalRelationSpec
        The ordered original typed operands.
    operation : str
        The supported binary transformation.
    shared_tapes : tuple of int or None
        One position per operand when using a single-tape join.
    keep_shared : bool
        Whether a join retains the selected tape at its left position.
    """

    left: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    right: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    operation: Literal["union", "concatenate", "intersect", "join"]
    shared_tapes: tuple[int, int] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    keep_shared: bool = False

    @dx.model_validator()
    def _validate_operation(self) -> None:
        """Check operand types, algebra, and supported single-tape parameters.

        Raises
        ------
        TypeError
            An operand is unchecked or tape types or algebras disagree.
        ValueError
            The operation has an invalid or absent required tape argument.
        """
        if not all(
            isinstance(source, RationalRelationSpec)
            for source in (self.left, self.right)
        ):
            raise TypeError("binary transforms require two checked relation operands")
        if self.left.algebra != self.right.algebra:
            raise TypeError("binary relation transforms require one shared algebra")
        if self.operation != "join" and (
            self.shared_tapes is not None or self.keep_shared
        ):
            raise ValueError("only join accepts shared tape positions and retention")
        left, right = self.left.topology, self.right.topology
        if self.operation in ("union", "concatenate"):
            if left.tapes != right.tapes or left.input_tapes != right.input_tapes:
                raise TypeError(
                    "relation expressions require identical ordered tape types"
                )
            return
        if self.operation == "intersect":
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
            supplied = self.shared_tapes
            if (
                not isinstance(supplied, tuple)
                or len(supplied) != 2
                or any(
                    type(i) is not int or not 0 <= i < len(source.tapes)
                    for i, source in zip(supplied, (left, right), strict=True)
                )
            ):
                raise ValueError("join requires one valid tape position per operand")
            l_tape, r_tape = supplied
        if left.tapes[l_tape] != right.tapes[r_tape]:
            raise TypeError("binary products require the same nominal shared tape")
        algebra = self.left._algebra
        if algebra is None or not algebra.establishes(
            *SEMIRING_LAWS, TIMES_COMMUTATIVE
        ):
            raise TypeError(
                "binary products require an established commutative semiring"
            )
        self._visible()

    def _visible(self) -> tuple[tuple[int, int], ...]:
        """Return ordered operand and tape positions for the result layout.

        Returns
        -------
        tuple
            The operand position and original tape for each result tape.
        """
        if self.operation in ("union", "concatenate"):
            return tuple((0, i) for i in range(len(self.left.tapes)))
        shared = (0, 0) if self.operation == "intersect" else self.shared_tapes
        assert shared is not None
        return _JoinLayout.build(
            self.left.topology,
            self.right.topology,
            *shared,
            keep_shared=self.keep_shared or self.operation == "intersect",
        ).visible

    def _validate_words(self, words: Words) -> None:
        """Check the observed words using their original typed tape positions.

        Parameters
        ----------
        words : Words
            The observed words on the result tapes, in order.

        Raises
        ------
        ValueError
            The word tuple does not inhabit the result tape layout.
        """
        visible = self._visible()
        if len(words) != len(visible):
            raise ValueError(
                "a transformed query needs exactly one word per retained tape"
            )
        padded: tuple[list[tuple[int, ...]], list[tuple[int, ...]]] = (
            [()] * len(self.left.tapes),
            [()] * len(self.right.tapes),
        )
        for (side, tape), word in zip(visible, words, strict=True):
            padded[side][tape] = word
        self.left.topology._validate_words(tuple(padded[0]))
        self.right.topology._validate_words(tuple(padded[1]))

    def bind(
        self,
        parameters: Mapping[ParameterId, torch.Tensor],
        *,
        max_states: int,
        max_work: int,
    ) -> RationalRelation:
        """Construct the checked sparse operation with its original leaves.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original shared scalar parameters of the two operands.
        max_states : int
            State limit for a sparse product.
        max_work : int
            Label-probe and arc examination limit for a sparse product.

        Returns
        -------
        RationalRelation
            The transformed sparse relation with live weight sources.
        """
        _validate_limits(max_states, max_work)
        left, right = _bind_relation_operands(
            self.left, self.right, parameters, description="binary relation"
        )
        match self.operation:
            case "union":
                return union(left, right)
            case "concatenate":
                return concatenate(left, right)
            case "intersect":
                return intersect(left, right, max_states=max_states, max_work=max_work)
            case "join":
                assert self.shared_tapes is not None
                return join(
                    left,
                    right,
                    left_tape=self.shared_tapes[0],
                    right_tape=self.shared_tapes[1],
                    keep_shared=self.keep_shared,
                    max_states=max_states,
                    max_work=max_work,
                )


class FixedTransformedRelationRequest(QiecModel):
    """A typed fixed observation of a transformed sparse relation.

    The wire preserves the original relation and the operation name.
    The provider creates the sparse topology and query once. Reads of
    the compiled query always evaluate the current original weights.

    Parameters
    ----------
    relation : UnaryRelationTransformSpec or BinaryRelationTransformSpec
        The original relation and its transformation.
    words : Words
        One observed word per transformed tape.
    max_states : int
        State budget for conditioning and each fixed observation.
    max_work : int
        Work budget for conditioning and each fixed observation.
    max_closure_size : int or None
        Optional bound for productive cyclic closure components.
    """

    relation: UnaryRelationTransformSpec | BinaryRelationTransformSpec = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int
    max_closure_size: int | None = None

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the result tape words, limits, and scalar algebra.

        Raises
        ------
        ValueError
            A word, limit, or algebra is invalid for a fixed relation query.
        """
        if not isinstance(
            self.relation, UnaryRelationTransformSpec | BinaryRelationTransformSpec
        ):
            raise TypeError("a transformed query requires a checked relation recipe")
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        self.relation._validate_words(self.words)
        if self._algebra not in (MARKOV.subject, LOG_PROB.subject):
            raise ValueError("fixed transformed relations require Markov or LogProb")

    @property
    def operation(self) -> Literal["pair_weight"]:
        """Return the fixed-tuple path sum objective.

        Returns
        -------
        str
            The checked pair-weight objective.
        """
        return "pair_weight"

    @property
    def result_type(self) -> TypeExpr:
        """Return the scalar score carrier of the transformed relation.

        Returns
        -------
        TypeExpr
            LogWeight for LogProb and Real for Markov.
        """
        return LOG_WEIGHT if self._algebra == LOG_PROB.subject else REAL

    @property
    def _algebra(self) -> ClaimSubject:
        """Return the canonical algebra identity shared by the recipe's operands.

        Returns
        -------
        ClaimSubject
            The original algebra claim identity.
        """
        relation = self.relation
        return (
            relation.source.algebra
            if isinstance(relation, UnaryRelationTransformSpec)
            else relation.left.algebra
        )

    def plan(self, parameters: Mapping[ParameterId, torch.Tensor]) -> FixedStringQuery:
        """Compile the bounded sparse query with original scalar weights.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The live original source weights.

        Returns
        -------
        FixedStringQuery
            The reusable fixed observation plan.
        """
        relation = self.relation.bind(
            parameters, max_states=self.max_states, max_work=self.max_work
        )
        return relation.query(
            self.words,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=self.max_closure_size,
        )


__all__ = [
    "BinaryRelationTransformSpec",
    "FixedTransformedRelationRequest",
    "UnaryRelationTransformSpec",
]
