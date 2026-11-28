"""Exact inference over a forest of weighted rational string factors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.objects import StringSet
from quivers.core.relations import RationalRelation, _weight_snapshot
from quivers.core.relations.joins import join
from quivers.core.relations.paths import _conditional_log_ratio, _require_normalizer
from quivers.core.relations.queries import RelationLimitError, _validate_limits
from quivers.core.relations.topology import Arc, Boundary, RelationTopology
from quivers.core.relations.transforms import project


@dataclass(frozen=True, slots=True)
class StringVariable:
    """A typed string variable, optionally observed as one word."""

    name: str
    tape: StringSet
    observed: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a string variable requires a nonempty name")
        if not isinstance(self.tape, StringSet):
            raise TypeError("a string variable requires a StringSet tape")
        if self.observed is None:
            return
        word = tuple(self.observed)
        if any(
            type(symbol) is not int or not 0 <= symbol < self.tape.alphabet.cardinality
            for symbol in word
        ):
            raise ValueError(f"invalid observed symbol for variable {self.name!r}")
        object.__setattr__(self, "observed", word)


@dataclass(frozen=True, slots=True)
class StringFactor:
    """A relation with one distinct named variable assigned to each tape."""

    variables: tuple[str, ...]
    relation: RationalRelation

    def __post_init__(self) -> None:
        names = tuple(self.variables)
        if len(names) != len(self.relation.topology.tapes):
            raise ValueError("a string factor requires one variable per tape")
        if len(names) != len(set(names)):
            raise ValueError(
                "a string factor cannot repeat a variable on distinct tapes"
            )
        object.__setattr__(self, "variables", names)


class _IndicatorWeights(nn.Module):
    def __init__(self, anchor: RationalRelation) -> None:
        super().__init__()
        self.source = anchor.module()
        self._anchor = anchor

    def forward(self) -> torch.Tensor:
        return self._anchor.algebra.unit_like((1,), self._anchor._weights())


@dataclass(frozen=True, slots=True)
class _TreeBounds:
    max_states: int
    max_work: int
    max_closure_size: int | None

    def __post_init__(self) -> None:
        if self.max_states is None or self.max_work is None:
            raise ValueError("tree inference requires explicit state and work limits")
        _validate_limits(self.max_states, self.max_work, self.max_closure_size)

    def product(
        self,
        left: RationalRelation,
        right: RationalRelation,
        left_tape: int = 0,
        *,
        keep: bool = False,
    ) -> RationalRelation:
        return join(
            left,
            right,
            left_tape=left_tape,
            right_tape=0,
            keep_shared=keep,
            max_states=self.max_states,
            max_work=self.max_work,
        )

    def one_word(
        self, variable: StringVariable, anchor: RationalRelation
    ) -> RationalRelation:
        word = variable.observed
        assert word is not None
        if len(word) + 1 > self.max_states:
            raise RelationLimitError("observed variable exceeds max_states")
        if len(word) > self.max_work:
            raise RelationLimitError("observed variable exceeds max_work")
        topology = RelationTopology(
            (variable.tape,),
            0,
            len(word) + 1,
            tuple(Arc(i, i + 1, (symbol,), 0) for i, symbol in enumerate(word)),
            (Boundary(0, 0),),
            (Boundary(len(word), 0),),
            ("unit",),
        )
        return RationalRelation(
            topology, _IndicatorWeights(anchor), algebra=anchor.algebra
        )


class TreeStringMarginal:
    """An exact marginal compiled by eliminating the other string variables.

    The unary relation represents the unnormalized mass in the queried
    variable's own connected component. The partition includes every
    component. Each evaluation reads the original weight sources afresh.
    """

    def __init__(
        self,
        relation: RationalRelation,
        other_components: tuple[RationalRelation, ...] = (),
        *,
        max_states: int,
        max_work: int,
        max_closure_size: int | None = None,
    ) -> None:
        if relation.algebra is not MARKOV and relation.algebra is not LOG_PROB:
            raise TypeError("exact tree marginals require Markov or LogProb algebra")
        components = (relation, *other_components)
        if any(
            len(component.topology.tapes) != 1
            or component.algebra is not relation.algebra
            for component in components
        ):
            raise TypeError(
                "tree marginals require unary relations with one shared algebra"
            )
        self._relation = relation
        self._bounds = _TreeBounds(max_states, max_work, max_closure_size)
        self._normalizers = tuple(
            component._total_query(max_states, max_work, max_closure_size)
            for component in components
        )

    @property
    def relation(self) -> RationalRelation:
        """The unnormalized unary relation for the queried component."""
        return self._relation

    def partition(self) -> torch.Tensor:
        """Return total mass in the factor algebra, with its scale."""
        algebra = self.relation.algebra
        with _weight_snapshot():
            masses = [query.weight() for query in self._normalizers]
        total = masses[0]
        for mass in masses[1:]:
            if total.dtype != mass.dtype or total.device != mass.device:
                raise ValueError(
                    "string factors require a common weight dtype and device"
                )
            total = algebra.tensor_op(total, mass)
        if not algebra.contains(total).item():
            raise OverflowError("string partition is outside the evaluation precision")
        return total

    def log_prob(self, word: tuple[int, ...]) -> torch.Tensor:
        """Return the log probability of one word under the full graph."""
        bounds = self._bounds
        plan = self.relation.query(
            (word,),
            max_states=bounds.max_states,
            max_work=bounds.max_work,
            max_closure_size=bounds.max_closure_size,
        )
        algebra = self.relation.algebra
        with _weight_snapshot():
            masses = [query.weight() for query in self._normalizers]
            for mass in masses[1:]:
                _require_normalizer(mass, algebra)
            score = plan.weight()
        return _conditional_log_ratio(score, masses[0], algebra)


type _Vertex = str | int


class StringFactorGraph:
    """A checked graph of typed variables and weighted rational factors.

    Factors share one Markov or LogProb algebra. Exact tree inference
    accepts forests whose intermediate relation joins and closure
    calculations meet the supplied limits.
    """

    def __init__(
        self, variables: tuple[StringVariable, ...], factors: tuple[StringFactor, ...]
    ) -> None:
        self._variables = {variable.name: variable for variable in variables}
        if len(self._variables) != len(variables):
            raise ValueError("string variable names must be distinct")
        if not factors:
            raise ValueError("a string factor graph requires at least one factor")
        self._factors = tuple(factors)
        algebra = factors[0].relation.algebra
        if algebra is not MARKOV and algebra is not LOG_PROB:
            raise TypeError("exact string factors require Markov or LogProb algebra")
        self._neighbours: dict[_Vertex, list[_Vertex]] = {
            name: [] for name in self._variables
        }
        for factor_id, factor in enumerate(factors):
            if factor.relation.algebra is not algebra:
                raise TypeError("string factors require one shared algebra")
            self._neighbours[factor_id] = list(factor.variables)
            for name, tape in zip(
                factor.variables, factor.relation.topology.tapes, strict=True
            ):
                variable = self._variables.get(name)
                if variable is None or variable.tape != tape:
                    raise TypeError(
                        f"unknown or mistyped variable {name!r} at factor {factor_id}"
                    )
                self._neighbours[name].append(factor_id)

    def _tree_order(
        self, start: str, visited: set[_Vertex]
    ) -> list[tuple[_Vertex, _Vertex | None]]:
        order: list[tuple[_Vertex, _Vertex | None]] = [(start, None)]
        visited.add(start)
        for node, parent in order:
            for child in self._neighbours[node]:
                if child == parent:
                    continue
                if child in visited:
                    raise ValueError(
                        "exact tree string inference requires an acyclic factor graph"
                    )
                visited.add(child)
                order.append((child, node))
        return order

    def _eliminate(
        self, order: list[tuple[_Vertex, _Vertex | None]], bounds: _TreeBounds
    ) -> RationalRelation:
        messages: dict[_Vertex, RationalRelation | None] = {}
        for node, parent in reversed(order):
            if isinstance(node, str):
                variable = self._variables[node]
                message = (
                    None
                    if variable.observed is None
                    else bounds.one_word(variable, self._factors[0].relation)
                )
                for child in self._neighbours[node]:
                    if child != parent:
                        incoming = messages.pop(child)
                        assert incoming is not None
                        message = (
                            incoming
                            if message is None
                            else bounds.product(message, incoming, keep=True)
                        )
            else:
                factor = self._factors[node]
                message = factor.relation
                for tape in reversed(range(len(factor.variables))):
                    child = factor.variables[tape]
                    if child == parent:
                        continue
                    incoming = messages.pop(child)
                    if incoming is None:
                        keep = tuple(
                            i for i in range(len(message.topology.tapes)) if i != tape
                        )
                        message = project(message, keep)
                    else:
                        message = bounds.product(message, incoming, tape)
            messages[node] = message

        root = order[0][0]
        marginal = messages[root]
        if marginal is None:
            raise ValueError(
                f"isolated string variable {root!r} has no finite normalizer"
            )
        return marginal

    def tree_marginal(
        self,
        variable: str,
        *,
        max_states: int,
        max_work: int,
        max_closure_size: int | None = None,
    ) -> TreeStringMarginal:
        """Compile exact factor-to-variable and variable-to-factor messages.

        Each edge of each component is used once toward its chosen
        root. The state and work limits apply separately to the sparse
        products and marginal path plans. A productive cycle within a
        relation still requires star and a local closure size limit.
        """
        if variable not in self._variables:
            raise ValueError(f"unknown string variable {variable!r}")
        bounds = _TreeBounds(max_states, max_work, max_closure_size)
        visited: set[_Vertex] = set()
        orders = [self._tree_order(variable, visited)]
        orders.extend(
            self._tree_order(name, visited)
            for name in self._variables
            if name not in visited
        )
        marginal, *others = (self._eliminate(order, bounds) for order in orders)
        return TreeStringMarginal(
            marginal,
            tuple(others),
            max_states=max_states,
            max_work=max_work,
            max_closure_size=max_closure_size,
        )


__all__ = ["StringFactor", "StringFactorGraph", "StringVariable", "TreeStringMarginal"]
