"""Log-space belief propagation on explicitly declared finite string supports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import log, log1p, prod
from types import MappingProxyType
from typing import Literal

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import FixedStringQuery, RationalRelation, _weight_snapshot
from quivers.core.relations.graph import StringFactorGraph
from quivers.core.relations.queries import RelationLimitError, _validate_limits

type _Word = tuple[int, ...]
type _Messages = tuple[tuple[torch.Tensor, ...], ...]
type _Occurrence = tuple[int, int]


@dataclass(frozen=True, slots=True)
class StringBeliefResult:
    """The marginal estimates and measured fixed-point residual of a sweep."""

    log_marginals: Mapping[str, torch.Tensor]
    status: Literal["converged", "limit"]
    sweeps: int
    message_residual: float
    log_messages: _Messages


@dataclass(frozen=True, slots=True)
class _FactorTable:
    variables: tuple[str, ...]
    relation: RationalRelation
    shape: tuple[int, ...]
    queries: tuple[FixedStringQuery, ...]

    def log_values(self) -> torch.Tensor:
        weights = self.relation._weights()
        algebra = self.relation.algebra
        table = torch.stack(
            [query._plan.evaluate(weights, algebra) for query in self.queries]
        ).reshape(self.shape)
        if not table.is_floating_point():
            raise TypeError(
                "finite string messages require floating-point factor weights"
            )
        if not bool(algebra.contains(table).all()):
            raise OverflowError(
                "a string factor table is outside the evaluation precision"
            )
        if algebra is MARKOV:
            positive = table > 0
            safe = torch.where(positive, table, torch.ones_like(table))
            table = torch.where(positive, safe.log(), -torch.inf)
        if not bool(LOG_PROB.contains(table).all()):
            raise OverflowError(
                "a string factor table is outside the evaluation precision"
            )
        return table


def _normalized(log_weights: torch.Tensor) -> torch.Tensor:
    normalizer = LOG_PROB.join(log_weights, dim=0)
    if not torch.isfinite(normalizer).item():
        raise ValueError("a string belief message has no positive finite normalizer")
    return log_weights - normalizer


def _outgoing(
    table: torch.Tensor, cavities: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    # Prefixes and suffixes use log products. They remain usable when a
    # cavity excludes a word, because no -infinity is subtracted.
    arity = len(cavities)
    broadcasts = [
        cavity.reshape((*((1,) * i), -1, *((1,) * (arity - i - 1))))
        for i, cavity in enumerate(cavities)
    ]
    prefixes = [table]
    for message in broadcasts[:-1]:
        prefixes.append(prefixes[-1] + message)
    suffix = table.new_zeros(())
    outgoing: list[torch.Tensor] = []
    for axis in reversed(range(arity)):
        scores = prefixes[axis] + suffix
        axes = tuple(i for i in range(arity) if i != axis)
        outgoing.append(LOG_PROB.join(scores, dim=axes) if axes else scores)
        suffix = suffix + broadcasts[axis]
    return tuple(reversed(outgoing))


class FiniteStringBeliefPlan:
    """Compile factor scores on explicit supports for loopy sum-product.

    This is a finite model supplied by the caller. The gate on entries
    limits its construction, and does not impose a string length cutoff.
    Scores remain live and are read once per original weight source in
    each run. Factor tables follow the graph's declared tape order.
    """

    def __init__(
        self,
        graph: StringFactorGraph,
        supports: Mapping[str, tuple[_Word, ...]],
        *,
        max_entries: int,
        max_states: int,
        max_work: int,
        max_closure_size: int | None = None,
    ) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("max_entries must be a nonnegative integer")
        if max_states is None or max_work is None:
            raise ValueError(
                "finite string messages require explicit state and work limits"
            )
        _validate_limits(max_states, max_work, max_closure_size)
        if set(supports) != graph._variables.keys():
            raise ValueError(
                "finite string supports must name every graph variable exactly once"
            )
        actual: dict[str, tuple[_Word, ...]] = {}
        for name, variable in graph._variables.items():
            words = tuple(tuple(word) for word in supports[name])
            if not words or len(words) != len(set(words)):
                raise ValueError(f"finite support for {name!r} requires distinct words")
            if any(
                type(symbol) is not int
                or not 0 <= symbol < variable.tape.alphabet.cardinality
                for word in words
                for symbol in word
            ):
                raise ValueError(
                    f"finite support for {name!r} contains an invalid symbol"
                )
            if variable.observed is not None:
                if variable.observed not in words:
                    raise ValueError(
                        f"finite support for {name!r} excludes its observation"
                    )
                words = (variable.observed,)
            actual[name] = words
        self._supports = MappingProxyType(actual)
        shapes = tuple(
            tuple(len(actual[name]) for name in factor.variables)
            for factor in graph._factors
        )
        if sum(prod(shape) for shape in shapes) > max_entries:
            raise RelationLimitError("finite string belief plan exceeds max_entries")
        occurrences: dict[str, list[_Occurrence]] = {name: [] for name in actual}
        factors = []
        for index, (factor, shape) in enumerate(
            zip(graph._factors, shapes, strict=True)
        ):
            for axis, name in enumerate(factor.variables):
                occurrences[name].append((index, axis))
            queries = []
            for words in product(*(actual[name] for name in factor.variables)):
                plan = factor.relation.query(
                    words,
                    max_states=max_states,
                    max_work=max_work,
                    max_closure_size=max_closure_size,
                )
                max_states -= plan.work.visited_states
                max_work -= plan.work.candidate_arcs
                queries.append(plan)
            factors.append(
                _FactorTable(factor.variables, factor.relation, shape, tuple(queries))
            )
        self._tables = tuple(factors)
        self._occurrences = occurrences

    @property
    def supports(self) -> Mapping[str, tuple[_Word, ...]]:
        """The word order along each variable's finite axis."""
        return self._supports

    def _variables(
        self, messages: _Messages, like: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[_Occurrence, torch.Tensor]]:
        beliefs = {}
        cavities = {}
        for name, incidence in self._occurrences.items():
            unit = like.new_zeros(len(self._supports[name]))
            prefixes = [unit]
            for f, axis in incidence:
                prefixes.append(prefixes[-1] + messages[f][axis])
            beliefs[name] = _normalized(prefixes[-1])
            suffix = unit
            for i, (f, axis) in reversed(tuple(enumerate(incidence))):
                cavities[f, axis] = _normalized(prefixes[i] + suffix)
                suffix = suffix + messages[f][axis]
        return beliefs, cavities

    def run(
        self, *, max_sweeps: int, tolerance: float | None = 1e-9, damping: float = 0.0
    ) -> StringBeliefResult:
        """Run synchronous log-space messages on the declared model.

        Damping mixes probabilities and lies in [0, 1). Convergence
        means the maximum undamped message total variation is within
        tolerance. The final residual is measured on the returned
        messages. A tolerance of None executes exactly the supplied
        number of sweeps. Autograd follows the executed schedule.
        """
        if type(max_sweeps) is not int or max_sweeps < 0:
            raise ValueError("max_sweeps must be a nonnegative integer")
        if not (
            (tolerance is None or 0 <= tolerance < float("inf")) and 0 <= damping < 1
        ):
            raise ValueError("invalid string message tolerance or damping")
        with _weight_snapshot():
            tables = tuple(factor.log_values() for factor in self._tables)
        like = tables[0]
        if any(
            table.dtype != like.dtype or table.device != like.device for table in tables
        ):
            raise ValueError(
                "string factor tables require a common weight dtype and device"
            )
        messages: _Messages = tuple(
            tuple(like.new_full((size,), -log(size)) for size in factor.shape)
            for factor in self._tables
        )
        sweeps = 0
        while True:
            beliefs, cavities = self._variables(messages, like)
            proposed = tuple(
                tuple(
                    map(
                        _normalized,
                        _outgoing(
                            table, tuple(cavities[f, i] for i in range(table.ndim))
                        ),
                    )
                )
                for f, table in enumerate(tables)
            )
            residual = max(
                0.5 * (old.exp() - new.exp()).abs().sum().item()
                for old_factor, new_factor in zip(messages, proposed, strict=True)
                for old, new in zip(old_factor, new_factor, strict=True)
            )
            converged = tolerance is not None and residual <= tolerance
            if converged or sweeps == max_sweeps:
                return StringBeliefResult(
                    MappingProxyType(beliefs),
                    "converged" if converged else "limit",
                    sweeps,
                    residual,
                    messages,
                )
            if damping:
                messages = tuple(
                    tuple(
                        LOG_PROB.join(
                            torch.stack((old + log(damping), new + log1p(-damping))),
                            dim=0,
                        )
                        for old, new in zip(old_factor, new_factor, strict=True)
                    )
                    for old_factor, new_factor in zip(messages, proposed, strict=True)
                )
            else:
                messages = proposed
            sweeps += 1


__all__ = ["FiniteStringBeliefPlan", "StringBeliefResult"]
