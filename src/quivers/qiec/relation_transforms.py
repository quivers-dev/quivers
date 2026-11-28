"""Checked recipes for unary operations on sparse rational relations."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.laws import ClaimSubject
from quivers.core.relations import FixedStringQuery, RationalRelation, Words
from quivers.core.relations.queries import _validate_limits
from quivers.core.relations.topology import _validate_tape_words
from quivers.qiec._models import QiecModel
from quivers.qiec._relation_operations import (
    _RelationInterface,
    _apply_binary,
    _apply_unary,
    _binary_layout,
    _unary_layout,
)
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.relation_cascades import RelationCascadeSpec
from quivers.qiec.relations import (
    RationalRelationSpec,
    _bind_relation_operands,
    RelationWeightBindings,
)
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
        self._layout()

    def _layout(self) -> _RelationInterface:
        """Return the checked tape layout and scalar algebra.

        Returns
        -------
        _RelationInterface
            The result's nominal tapes, orientation, and scalar algebra.
        """
        return _unary_layout(
            _RelationInterface.of(self.source),
            self.operation,
            self.positions,
            self.word,
        )

    def _validate_words(self, words: Words) -> None:
        """Validate observations against the transformed tape layout.

        Parameters
        ----------
        words : Words
            The observed word on each result tape.

        Raises
        ------
        ValueError
            The word tuple has the wrong arity or an invalid symbol.
        """
        tapes = self._layout().tapes
        if len(words) != len(tapes):
            raise ValueError(
                "a transformed query needs exactly one word per retained tape"
            )
        _validate_tape_words(tapes, words)

    def bind(
        self,
        parameters: RelationWeightBindings,
        *,
        max_states: int,
        max_work: int,
    ) -> RationalRelation:
        """Construct and bind the transformed sparse relation.

        Parameters
        ----------
        parameters : RelationWeightBindings
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
        return _apply_unary(
            source,
            self.operation,
            self.positions,
            self.word,
            max_states=max_states,
            max_work=max_work,
        )


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
        self._layout()

    def _layout(self) -> _RelationInterface:
        """Return the checked ordered output tapes and scalar algebra.

        Returns
        -------
        _RelationInterface
            The result's nominal tapes, orientation, and scalar algebra.
        """
        return _binary_layout(
            _RelationInterface.of(self.left),
            _RelationInterface.of(self.right),
            self.operation,
            self.shared_tapes,
            self.keep_shared,
        )

    def _validate_words(self, words: Words) -> None:
        """Validate observations against the transformed tape layout.

        Parameters
        ----------
        words : Words
            The observed word on each result tape.

        Raises
        ------
        ValueError
            The word tuple has the wrong arity or an invalid symbol.
        """
        tapes = self._layout().tapes
        if len(words) != len(tapes):
            raise ValueError(
                "a transformed query needs exactly one word per retained tape"
            )
        _validate_tape_words(tapes, words)

    def bind(
        self,
        parameters: RelationWeightBindings,
        *,
        max_states: int,
        max_work: int,
    ) -> RationalRelation:
        """Construct the checked sparse operation with its original leaves.

        Parameters
        ----------
        parameters : RelationWeightBindings
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
        return _apply_binary(
            left,
            right,
            self.operation,
            self.shared_tapes,
            self.keep_shared,
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
    relation : UnaryRelationTransformSpec or BinaryRelationTransformSpec or RelationCascadeSpec
        The original relation and its transformation.
    words : Words
        One observed word per transformed tape.
    max_states : int
        State budget for conditioning and each fixed observation.
    max_work : int
        Work budget for conditioning and each fixed observation. A cascade
        also limits the number of steps and the records in each topology.
    max_closure_size : int or None
        Optional bound for productive cyclic closure components.
    """

    relation: (
        UnaryRelationTransformSpec | BinaryRelationTransformSpec | RelationCascadeSpec
    ) = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
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
            self.relation,
            UnaryRelationTransformSpec
            | BinaryRelationTransformSpec
            | RelationCascadeSpec,
        ):
            raise TypeError("a transformed query requires a checked relation recipe")
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        if isinstance(self.relation, RelationCascadeSpec):
            _validate_tape_words(self.relation._layout.tapes, self.words)
        else:
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
        if isinstance(relation, RelationCascadeSpec):
            return relation.algebra
        return (
            relation.source.algebra
            if isinstance(relation, UnaryRelationTransformSpec)
            else relation.left.algebra
        )

    def plan(self, parameters: RelationWeightBindings) -> FixedStringQuery:
        """Compile the bounded sparse query with original scalar weights.

        Parameters
        ----------
        parameters : RelationWeightBindings
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
