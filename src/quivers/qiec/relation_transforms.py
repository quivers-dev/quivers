"""Checked recipes for unary operations on sparse rational relations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    FixedStringQuery,
    RationalRelation,
    Words,
    closure,
    condition,
    invert,
    project,
    trim,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec
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


class FixedTransformedRelationRequest(QiecModel):
    """A typed fixed observation of a transformed sparse relation.

    The wire preserves the original relation and the operation name.
    The provider creates the sparse topology and query once. Reads of
    the compiled query always evaluate the current original weights.

    Parameters
    ----------
    relation : UnaryRelationTransformSpec
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

    relation: UnaryRelationTransformSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
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
        if not isinstance(self.relation, UnaryRelationTransformSpec):
            raise TypeError(
                "a transformed query requires a checked unary relation recipe"
            )
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        self.relation._validate_words(self.words)
        if self.relation.source.algebra not in (MARKOV.subject, LOG_PROB.subject):
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
        return LOG_WEIGHT if self.relation.source.algebra == LOG_PROB.subject else REAL

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


__all__ = ["FixedTransformedRelationRequest", "UnaryRelationTransformSpec"]
