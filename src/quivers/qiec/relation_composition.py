"""Typed stable composition of two sparse QIEC relation specifications."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    ComposedRelation,
    ComposedStringQuery,
    FixedStringQuery,
    Words,
)
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec._relation_operations import _RelationInterface, _binary_layout
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.contracts import ParameterId
from quivers.qiec.relations import RationalRelationSpec, _bind_relation_operands
from quivers.qiec.types import REAL, TypeExpr


class ComposedRelationSpec(QiecModel):
    """A checked binary composition that keeps both original topologies.

    Each operand has one input and one output tape. The left output
    and right input must have the same nominal type. Both operands
    have the same trusted Markov or LogProb algebra.

    Parameters
    ----------
    left : RationalRelationSpec
        The source-side operand.
    right : RationalRelationSpec
        The target-side operand.
    """

    left: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    right: RationalRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_operands(self) -> None:
        """Check tape arities, the shared tape, and the common scalar algebra.

        Raises
        ------
        ValueError
            If operand types or scalar algebras do not match the composition.
        """
        _binary_layout(
            _RelationInterface.of(self.left),
            _RelationInterface.of(self.right),
            "compose",
            None,
            False,
        )

    def bind(self, parameters: Mapping[ParameterId, torch.Tensor]) -> ComposedRelation:
        """Bind the original identities and construct a sparse composition.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            One explicit mapping for scalar weights of both operands.

        Returns
        -------
        ComposedRelation
            The checked composition, with the original live parameters.

        Raises
        ------
        TypeError
            If an operand cannot bind the proposed tensors.
        ValueError
            If different identities alias one tensor or the bindings have
            different dtypes or devices.
        """
        left, right = _bind_relation_operands(
            self.left, self.right, parameters, description="composition"
        )
        return ComposedRelation(left, right)


class FixedComposedRelationRequest(QiecModel):
    """A checked fixed outer-word composition score and explicit work limits.

    The wire keeps separate operands and their original weight owners.
    Lazy scoring constructs a fixed-word product. Materialized scoring
    constructs a reusable sparse product and can use explicitly permitted
    productive component closure.

    Parameters
    ----------
    relation : ComposedRelationSpec
        The algebra and the two typed operand topologies.
    words : Words
        The source input word and target output word in that order.
    execution : str
        Select lazy fixed-word or materialized sparse product.
    max_states : int
        State limits for each product construction and word query.
    max_work : int
        Work limits for each product construction and word query.
    max_closure_size : int or None
        The productive closure bound for a materialized word query.
    """

    relation: ComposedRelationSpec = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    words: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_states: int
    max_work: int
    execution: Literal["lazy", "materialized"] = "lazy"
    max_closure_size: int | None = None

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check the outer observation and the explicit execution limits.

        Raises
        ------
        ValueError
            If the word tuple or any limit is invalid, or lazy execution
            requests closure.
        """
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)
        if len(self.words) != 2:
            raise ValueError("composition requires exactly two outer words")
        self.relation.left.topology._validate_words((self.words[0], ()))
        self.relation.right.topology._validate_words(((), self.words[1]))
        if self.execution == "lazy" and self.max_closure_size is not None:
            raise ValueError(
                "a composition closure budget requires materialized execution"
            )

    def plan(
        self, parameters: Mapping[ParameterId, torch.Tensor]
    ) -> ComposedStringQuery | FixedStringQuery:
        """Compile the requested sparse plan with its original weight owners.

        Parameters
        ----------
        parameters : Mapping of ParameterId to torch.Tensor
            The original scalar weights shared by the two operands.

        Returns
        -------
        ComposedStringQuery or FixedStringQuery
            The checked lazy or materialized fixed-word query.
        """
        relation = self.relation.bind(parameters)
        if self.execution == "lazy":
            return relation.query(
                self.words, max_states=self.max_states, max_work=self.max_work
            )
        materialized = relation.materialize(
            max_states=self.max_states, max_work=self.max_work
        )
        return materialized.query(
            self.words,
            max_states=self.max_states,
            max_work=self.max_work,
            max_closure_size=self.max_closure_size,
        )

    @property
    def operation(self) -> Literal["pair_weight"]:
        """Return the scalar objective recognized by the QIEC provider.

        Returns
        -------
        str
            The pair-weight objective.
        """
        return "pair_weight"

    @property
    def result_type(self) -> TypeExpr:
        """Return the checked scalar type of the composed pair weight.

        Returns
        -------
        TypeExpr
            LogWeight for LogProb and Real for Markov.
        """
        return LOG_WEIGHT if self.relation.left.algebra == LOG_PROB.subject else REAL


__all__ = ["ComposedRelationSpec", "FixedComposedRelationRequest"]
