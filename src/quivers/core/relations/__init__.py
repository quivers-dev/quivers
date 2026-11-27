"""Sparse weighted relations over strings and finite observations of them."""

from __future__ import annotations

from functools import cached_property
from itertools import product
from typing import cast

import torch
from torch import nn

from quivers.core.algebras import REAL, Semiring
from quivers.core.morphisms import FiniteMorphism, Morphism
from quivers.core.objects import FreeMonoid, ProductSet, SetObject, Unit
from quivers.core.relations.queries import (
    ProductiveEpsilonCycleError,
    QueryWork,
    RelationLimitError,
    _compile_paths,
    _PathPlan,
    _validate_limits,
)
from quivers.core.relations.topology import (
    EPSILON,
    Arc,
    Boundary,
    Label,
    RelationTopology,
    Words,
)


__all__ = [
    "EPSILON",
    "Arc",
    "Boundary",
    "FixedStringQuery",
    "FiniteRelationRestriction",
    "Label",
    "ProductiveEpsilonCycleError",
    "QueryWork",
    "RationalRelation",
    "RelationLimitError",
    "RelationTopology",
    "Words",
]


def _tape_object(tapes: tuple[SetObject, ...]) -> SetObject:
    if not tapes:
        return Unit
    if len(tapes) == 1:
        return tapes[0]
    return ProductSet(components=tapes)


class _TensorWeights(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        if isinstance(weights, nn.Parameter):
            self.register_parameter("weights", weights)
        elif weights.is_leaf:
            self.register_buffer("weights", weights)
        else:
            raise TypeError("a computed weight tensor requires a weight source module")

    def forward(self) -> torch.Tensor:
        return cast(torch.Tensor, self.weights)


class RationalRelation(Morphism):
    """A finite-state representation of a weighted relation on ordered tapes.

    At every fixed tuple of words, the relation sums the weights of all
    accepting walks realizing that tuple. A walk's weight multiplies its
    initial weight, its arc weights in traversal order, and its final
    weight. An empty walk contributes its initial and final weights.
    Topology and parameter values are separate.

    Parameters
    ----------
    topology : RelationTopology
        Finite topology and weight-slot names.
    weights : torch.Tensor or nn.Module
        A vector of scalar weights in slot order. A tensor must be a leaf.
        A module supplies the vector through its zero-argument forward call
        and can compute fresh differentiable weights from its parameters.
    algebra : Semiring
        The scalar algebra for path multiplication and addition. Defaults
        to real weights.

    Notes
    -----
    Fixed-word queries accept cycles that consume symbols, because each
    such step advances a position in a finite observation. A reachable
    all-epsilon cycle with an accepting continuation instead requires a
    star calculation. This evaluator refuses that case, even when its
    current weights are zero. A change in weights never changes the plan.
    """

    def __init__(
        self,
        topology: RelationTopology,
        weights: torch.Tensor | nn.Module,
        *,
        algebra: Semiring = REAL,
    ) -> None:
        if not isinstance(algebra, Semiring) or algebra.element_shape:
            raise TypeError("a rational relation requires a scalar semiring")
        super().__init__(
            _tape_object(topology.tapes[: topology.input_tapes]),
            _tape_object(topology.tapes[topology.input_tapes :]),
            algebra,
        )
        self._topology = topology
        if isinstance(weights, torch.Tensor):
            self._check_weights(weights)
            self._weight_source = _TensorWeights(weights)
        elif isinstance(weights, nn.Module):
            self._weight_source = weights
        else:
            raise TypeError("weights must be a tensor or a weight source module")

    @property
    def topology(self) -> RelationTopology:
        """The immutable topology of this relation."""
        return self._topology

    @property
    def algebra(self) -> Semiring:
        return cast(Semiring, super().algebra)

    def module(self) -> nn.Module:
        """The original weight source and its parameters."""
        return self._weight_source

    def _check_weights(self, weights: torch.Tensor) -> None:
        expected = (len(self._topology.weight_names),)
        if not isinstance(weights, torch.Tensor) or weights.shape != expected:
            raise ValueError(
                f"the weight source must supply a tensor of shape {expected}"
            )

    def _weights(self) -> torch.Tensor:
        values = self._weight_source()
        self._check_weights(values)
        return values

    def query(
        self,
        words: Words,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
    ) -> FixedStringQuery:
        """Compile a reusable query for one tuple of words.

        Parameters
        ----------
        words : Words
            Observations in tape order. Each word contains only symbols
            of its tape's alphabet.
        max_states : int or None
            Limit on visited state and offset configurations.
        max_work : int or None
            Limit on candidate arc examinations, including rejected matches.

        Returns
        -------
        FixedStringQuery
            The query and construction counts.

        Raises
        ------
        RelationLimitError
            If a construction limit is reached before the query is ready.
        ProductiveEpsilonCycleError
            If there are infinitely many structurally accepting walks at
            the queried tuple.
        """
        return FixedStringQuery(self, words, max_states=max_states, max_work=max_work)

    def weight(self, words: Words) -> torch.Tensor:
        """Evaluate the total accepting walk weight for the supplied words.

        Parameters
        ----------
        words : Words
            One observed word per tape, in tape order.

        Returns
        -------
        torch.Tensor
            The scalar in the relation's algebra.
        """
        return self.query(words).weight()

    def restrict(
        self,
        max_lengths: tuple[int, ...],
        *,
        max_entries: int,
        max_states: int | None = None,
        max_work: int | None = None,
    ) -> FiniteRelationRestriction:
        """View a declared length restriction as a finite tensor morphism.

        Parameters
        ----------
        max_lengths : tuple of int
            Inclusive maximum lengths in tape order.
        max_entries : int
            Maximum number of tensor entries, checked before allocation.
        max_states : int or None
            Total product states that may be visited across the bounded table.
        max_work : int or None
            Total candidate arc examinations across the bounded table.

        Returns
        -------
        FiniteRelationRestriction
            A differentiable view over the bounded string carriers.
        """
        return FiniteRelationRestriction(
            self,
            max_lengths,
            max_entries=max_entries,
            max_states=max_states,
            max_work=max_work,
        )


class FixedStringQuery:
    """A path plan with live access to the relation's weights.

    Parameters
    ----------
    relation : RationalRelation
        The relation of the query.
    words : Words
        One word per tape in tape order.
    max_states : int or None
        Limit on visited state and offset configurations.
    max_work : int or None
        Limit on candidate arc examinations.
    """

    def __init__(
        self,
        relation: RationalRelation,
        words: Words,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
    ) -> None:
        relation.topology._validate_words(words)
        _validate_limits(max_states, max_work)
        self._plan = _compile_paths(
            relation.topology, words, max_states=max_states, max_work=max_work
        )
        self._relation = relation

    @property
    def work(self) -> QueryWork:
        """Work of compiling this query."""
        return self._plan.work

    def weight(self) -> torch.Tensor:
        """Evaluate the plan with the current weights."""
        return self._plan.evaluate(self._relation._weights(), self._relation.algebra)


class FiniteRelationRestriction(FiniteMorphism):
    """A size-checked finite tensor view of a rational relation.

    Parameters
    ----------
    relation : RationalRelation
        The relation to restrict.
    max_lengths : tuple of int
        Inclusive bounds for tape lengths, in tape order.
    max_entries : int
        Maximum number of entries in the resulting tensor.
    max_states : int or None
        Total product states visited across the bounded table.
    max_work : int or None
        Total candidate arc examinations across the bounded table.
    """

    def __init__(
        self,
        relation: RationalRelation,
        max_lengths: tuple[int, ...],
        *,
        max_entries: int,
        max_states: int | None = None,
        max_work: int | None = None,
    ) -> None:
        _validate_limits(max_states, max_work)
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if len(max_lengths) != len(relation.topology.tapes):
            raise ValueError("a restriction must bound every tape")
        if any(type(length) is not int or length < 0 for length in max_lengths):
            raise ValueError("every tape length bound must be a nonnegative integer")
        remaining = max_entries
        carriers: list[FreeMonoid] = []
        for tape, length in zip(relation.topology.tapes, max_lengths):
            try:
                bounded = tape.bounded(length, max_size=remaining)
            except ValueError as error:
                raise RelationLimitError(
                    "finite restriction exceeds max_entries"
                ) from error
            carriers.append(bounded)
            remaining //= bounded.size
        input_tapes = relation.topology.input_tapes
        super().__init__(
            _tape_object(tuple(carriers[:input_tapes])),
            _tape_object(tuple(carriers[input_tapes:])),
            relation.algebra,
        )
        self._carriers = tuple(carriers)
        self._relation = relation
        self._max_states = max_states
        self._max_work = max_work

    def module(self) -> nn.Module:
        """The original relation's weight source."""
        return self._relation.module()

    @cached_property
    def _plans(self) -> tuple[_PathPlan, ...]:
        state_budget = self._max_states
        work_budget = self._max_work
        plans: list[_PathPlan] = []
        for indexes in product(*(range(carrier.size) for carrier in self._carriers)):
            words = tuple(
                carrier.decode(index) for carrier, index in zip(self._carriers, indexes)
            )
            plan = _compile_paths(
                self._relation.topology,
                words,
                max_states=state_budget,
                max_work=work_budget,
            )
            plans.append(plan)
            if state_budget is not None:
                state_budget -= plan.work.visited_states
            if work_budget is not None:
                work_budget -= plan.work.candidate_arcs
        return tuple(plans)

    @property
    def tensor(self) -> torch.Tensor:
        """Evaluate the bounded table using a fresh weight-source forward call."""
        plans = self._plans
        values = self._relation._weights()
        algebra = self._relation.algebra
        return torch.stack([plan.evaluate(values, algebra) for plan in plans]).reshape(
            self.tensor_shape
        )
