"""Sparse weighted relations over strings and finite observations of them."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import cached_property
from itertools import product
from typing import cast

import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV, REAL, Semiring
from quivers.core.morphisms import FiniteMorphism, Morphism
from quivers.core.objects import FreeMonoid, ProductSet, SetObject, Unit
from quivers.core.relations.paths import (
    ArcCounts,
    PathStatistics,
    RelationPath,
    _PathWalks,
    _markov_log_weights,
    _probability_algebra,
    _require_normalizer,
)
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
    "ArcCounts",
    "Boundary",
    "ComposedRelation",
    "ComposedStringQuery",
    "ConditionalStringQuery",
    "ContextualEditChannel",
    "ContextualEditPlan",
    "FixedStringQuery",
    "FailureArc",
    "FiniteRelationRestriction",
    "FiniteStringBeliefPlan",
    "FiniteStringScore",
    "Label",
    "MarginalOutputChoice",
    "NgramFamily",
    "NgramMessagePlan",
    "NgramProjection",
    "PenalizedNgramProjection",
    "PathStatistics",
    "ProductiveEpsilonCycleError",
    "QueryWork",
    "RationalRelation",
    "RegularNgramProjection",
    "RegularStringBelief",
    "RegularStringEPResult",
    "RegularStringSite",
    "RegularStringSiteUpdate",
    "RelationLimitError",
    "RelationPath",
    "RelationTopology",
    "StringAlignment",
    "StringAlignmentCertificate",
    "StringAlignmentMapResult",
    "StringBeliefResult",
    "StringEPResult",
    "StringSiteProjection",
    "StringFactor",
    "StringFactorGraph",
    "StringMapCertificate",
    "StringMapResult",
    "StringCandidateMessage",
    "StringCandidatePlan",
    "StringVariable",
    "TreeStringMarginal",
    "Words",
    "auto_intersect",
    "backoff_query",
    "backoff_total_weight",
    "best_marginal_output",
    "closure",
    "compile_ranked_ot",
    "concatenate",
    "condition",
    "dual_alignment_map",
    "dual_string_map",
    "intersect",
    "invert",
    "join",
    "penalized_regular_ngram_projection",
    "minimize_static_acceptor",
    "project",
    "regular_string_ep",
    "ranked_output_candidates",
    "trim",
    "union",
]


_weight_cache: ContextVar[dict[int, torch.Tensor] | None] = ContextVar(
    "_weight_cache", default=None
)


@contextmanager
def _weight_snapshot() -> Iterator[None]:
    token = _weight_cache.set({})
    try:
        yield
    finally:
        _weight_cache.reset(token)


def _read_weight_source(source: nn.Module) -> torch.Tensor:
    snapshot = _weight_cache.get()
    if snapshot is None:
        return source()
    key = id(source)
    if key not in snapshot:
        snapshot[key] = source()
    return snapshot[key]


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
        A vector of scalar weights in slot order, with every entry in the
        declared algebra's carrier. A tensor must be a leaf. A module
        supplies the vector through its zero-argument forward call and
        can compute fresh differentiable weights from its parameters.
    algebra : Semiring
        The scalar algebra for path multiplication and addition. Defaults
        to real weights.

    Notes
    -----
    Fixed-word queries accept cycles that consume symbols, because each
    such step advances a position in a finite observation. A reachable
    all-epsilon cycle with an accepting continuation requires a star
    calculation. Supplying `max_closure_size` to `query` admits productive
    cyclic components under an established semiring star. The plan
    depends on topology and records zero-weight
    occurrences. Each evaluation checks the current weight vector and
    the numerical domain of every closure it uses.
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
        self.algebra.validate(weights)

    def _weights(self) -> torch.Tensor:
        values = _read_weight_source(self._weight_source)
        self._check_weights(values)
        return values

    def query(
        self,
        words: Words,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
        max_closure_size: int | None = None,
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
        max_closure_size : int or None
            Maximum order of each local star matrix, including auxiliary
            vertices for parallel arcs. None requires an acyclic
            productive graph.

        Returns
        -------
        FixedStringQuery
            The query and construction counts.

        Raises
        ------
        RelationLimitError
            If a construction limit is reached before the query is ready.
        ProductiveEpsilonCycleError
            If the productive graph has a cycle and closure was not requested.
        """
        return FixedStringQuery(
            self,
            words,
            max_states=max_states,
            max_work=max_work,
            max_closure_size=max_closure_size,
        )

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

    def conditional(
        self,
        input_words: Words,
        *,
        max_states: int,
        max_work: int,
        max_closure_size: int | None = None,
    ) -> ConditionalStringQuery:
        """Compile a conditional over outputs for the fixed input words."""
        return ConditionalStringQuery(
            self,
            input_words,
            max_states=max_states,
            max_work=max_work,
            max_closure_size=max_closure_size,
        )

    def total_weight(
        self,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
        max_closure_size: int | None = None,
    ) -> torch.Tensor:
        """Sum the weights of all accepting walks over every word tuple.

        One additional linear pass erases the tape labels. The usual
        state, arc examination and closure limits then apply. A
        productive cycle requires the established star to decide a
        usable weight for its component.
        """
        return self._total_query(max_states, max_work, max_closure_size).weight()

    def _total_query(
        self,
        max_states: int | None,
        max_work: int | None,
        max_closure_size: int | None,
    ) -> FixedStringQuery:
        topology = self.topology
        labels = (EPSILON,) * len(topology.tapes)
        erased = replace(
            topology, arcs=tuple(replace(arc, labels=labels) for arc in topology.arcs)
        )
        relation = RationalRelation(erased, self.module(), algebra=self.algebra)
        return relation.query(
            ((),) * len(topology.tapes),
            max_states=max_states,
            max_work=max_work,
            max_closure_size=max_closure_size,
        )

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
    max_closure_size : int or None
        Maximum order of each local star matrix. None requires an
        acyclic productive graph.
    """

    def __init__(
        self,
        relation: RationalRelation,
        words: Words,
        *,
        max_states: int | None = None,
        max_work: int | None = None,
        max_closure_size: int | None = None,
    ) -> None:
        relation.topology._validate_words(words)
        _validate_limits(max_states, max_work, max_closure_size)
        self._plan = _compile_paths(
            relation.topology,
            words,
            max_states=max_states,
            max_work=max_work,
            max_closure_size=max_closure_size,
        )
        self._relation = relation
        self._words = tuple(tuple(word) for word in words)
        self._max_states = max_states
        self._max_work = max_work
        self._max_closure_size = max_closure_size

    @cached_property
    def _walks(self) -> _PathWalks:
        return _PathWalks(
            self._relation.topology,
            self._words,
            self._max_states,
            self._max_work,
            self._max_closure_size,
        )

    @property
    def work(self) -> QueryWork:
        """Work of compiling this query."""
        return self._plan.work

    def weight(self, *, recompute_backward: bool = False) -> torch.Tensor:
        """Evaluate the plan with the current weights.

        A cyclic plan requires floating-point weights and an algebra that
        establishes the semiring laws and the least-solution law of its
        star. Closure can raise `DivergentClosureError` or
        `UndecidedClosureError` for the current weights.

        With `recompute_backward`, autograd retains the path-plan inputs
        and recomputes required intermediate values during backward.
        The weight source runs once, outside the checkpoint.
        """
        weights = self._relation._weights()
        algebra = self._relation.algebra
        if recompute_backward and torch.is_grad_enabled() and weights.requires_grad:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                self._plan.evaluate, weights, algebra, use_reentrant=False
            )
        return self._plan.evaluate(weights, algebra)

    def best_path(self) -> RelationPath | None:
        """Return the highest-weight positive path, if the query has one.

        The productive graph must be acyclic, and weights must belong
        to Markov or LogProb. The result identifies a single path.
        """
        paths = self.k_best_paths(1)
        return paths[0] if paths else None

    def best_path_weight(self) -> torch.Tensor:
        """Return the best positive path weight, or the semiring zero.

        The same exact ranking and deterministic tie order as
        `best_path` select the path. The result differentiates its
        original occurrences with the selected path held fixed. At a
        tie the derivative is that of the selected path. If no positive
        path exists, the result is a constant semiring zero.
        """
        weights = self._relation._weights()
        algebra = self._relation.algebra
        paths = self._walks.k_best(1, weights, algebra)
        return paths[0].weight if paths else algebra.zero_like((), weights)

    def k_best_paths(self, k: int) -> tuple[RelationPath, ...]:
        """Return up to `k` positive paths in decreasing weight order.

        Ties follow a deterministic order determined by the topology.
        Ranking compares the exact rational values of the stored binary
        weights. Each arc supplies a ranked stream of suffixes,
        and merging computes only the heads needed for the result.
        Zero-weight paths are omitted.
        """
        if type(k) is not int or k < 0:
            raise ValueError("k must be a nonnegative integer")
        return self._walks.k_best(k, self._relation._weights(), self._relation.algebra)

    @property
    def max_path_length(self) -> int | None:
        """Return the maximum arc count over structurally accepting paths.

        The productive graph must be acyclic. A graph without accepting
        paths has no maximum. The bound includes paths whose current
        weights are zero, since subsequent weight changes may enable them.
        """
        return self._walks.max_path_length

    def path_log_prob(
        self, initial_state: int, arc_ids: tuple[int, ...], final_state: int
    ) -> torch.Tensor:
        """Score an original trace under the fixed-word path posterior.

        The trace supplies its original initial state, ordered arc
        occurrence IDs, and original final state. Invalid or zero-weight
        traces have log probability negative infinity. The observed path
        mass must be positive and finite. The positive trace score and
        the normalizer use the same current weight source.
        """
        log_weights = self._log_path_weights()
        total = self._path_log_normalizer(log_weights)
        return self._walks.path_log_prob(
            self._words, initial_state, arc_ids, final_state, log_weights, total
        )

    def _path_log_normalizer(self, log_weights: torch.Tensor) -> torch.Tensor:
        total = self._plan.evaluate(log_weights, LOG_PROB)
        _require_normalizer(total, LOG_PROB)
        return total

    def _log_path_weights(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        weights = self._relation._weights()
        algebra = _probability_algebra(weights, self._relation.algebra)
        weights = weights if dtype is None else weights.to(dtype=dtype)
        return _markov_log_weights(weights) if algebra is MARKOV else weights

    def sample_path(
        self, *, generator: torch.Generator | None = None, max_steps: int | None = None
    ) -> RelationPath:
        """Draw a path conditional on the observed words.

        Sampling uses float64 log-space suffix weights and a CPU
        generator. The observed Markov or LogProb mass must be
        positive and finite. A cyclic plan requires a supported
        closure and an explicit max_steps. An attempted traversal
        beyond that arc limit raises a resource error.
        The sampled path retains its original arc occurrences.
        """
        return self._walks.sample(
            self._relation._weights(), self._relation.algebra, generator, max_steps
        )

    def expected_arc_counts(self) -> ArcCounts:
        """Return differentiable conditional expected arc traversal counts.

        Two packed passes sum the prefixes and suffixes at each
        configuration. Occurrences of an original arc at different
        configurations are added after normalizing. The productive
        graph can use an explicitly permitted, finite closure. The
        Markov or LogProb normalizer must be positive and finite.
        """
        return self._walks.counts(self._relation._weights(), self._relation.algebra)

    def path_statistics(
        self,
        *,
        initial_costs: torch.Tensor | None = None,
        arc_costs: torch.Tensor | None = None,
        final_costs: torch.Tensor | None = None,
    ) -> PathStatistics:
        """Return path entropy and the expectation of optional additive costs.

        The observed Markov or LogProb mass must be positive and
        finite. Its normalization is evaluated in log space, with
        explicit closure support for a cyclic observation. Costs have one row
        per original mark or arc, in topology order. Boundary and
        arc costs can be combined when their trailing shapes agree.
        Derivatives on the current positive support flow through
        the weights and costs. Probability-zero choices are masked.
        """
        return self._walks.statistics(
            self._relation._weights(),
            self._relation.algebra,
            initial_costs,
            arc_costs,
            final_costs,
        )


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


from quivers.core.relations.calculus import (  # noqa: E402
    ComposedRelation,
    ComposedStringQuery,
)
from quivers.core.relations.expressions import closure, concatenate, union  # noqa: E402
from quivers.core.relations.joins import join  # noqa: E402
from quivers.core.relations.transforms import (  # noqa: E402
    condition,
    intersect,
    invert,
    project,
    trim,
)
from quivers.core.relations.graph import (  # noqa: E402
    StringFactor,
    StringFactorGraph,
    StringVariable,
    TreeStringMarginal,
)
from quivers.core.relations.conditional import ConditionalStringQuery  # noqa: E402
from quivers.core.relations.dual import (  # noqa: E402
    FiniteStringScore,
    StringMapCertificate,
    StringMapResult,
    dual_string_map,
)
from quivers.core.relations.projection import (  # noqa: E402
    NgramFamily,
    NgramProjection,
    PenalizedNgramProjection,
)
from quivers.core.relations.beliefs import (  # noqa: E402
    FiniteStringBeliefPlan,
    StringBeliefResult,
    StringEPResult,
    StringSiteProjection,
)
from quivers.core.relations.contextual_edit import (  # noqa: E402
    ContextualEditChannel,
    ContextualEditPlan,
)

from quivers.core.relations.alignment_dual import (  # noqa: E402
    StringAlignment,
    StringAlignmentCertificate,
    StringAlignmentMapResult,
    dual_alignment_map,
)

from quivers.core.relations.marginal_map import (  # noqa: E402
    MarginalOutputChoice,
    best_marginal_output,
)
from quivers.core.relations.ngram_message import NgramMessagePlan  # noqa: E402

from quivers.core.relations.regular_projection import (  # noqa: E402
    RegularNgramProjection,
    penalized_regular_ngram_projection,
)

from quivers.core.relations.regular_ep import (  # noqa: E402
    RegularStringBelief,
    RegularStringEPResult,
    RegularStringSite,
    RegularStringSiteUpdate,
    regular_string_ep,
)
from quivers.core.relations.equality import auto_intersect  # noqa: E402
from quivers.core.relations.backoff import (  # noqa: E402
    FailureArc,
    backoff_query,
    backoff_total_weight,
)
from quivers.core.relations.minimization import minimize_static_acceptor  # noqa: E402
from quivers.core.relations.ranked import (  # noqa: E402
    compile_ranked_ot,
    ranked_output_candidates,
)
from quivers.core.relations.messages import (  # noqa: E402
    StringCandidateMessage,
    StringCandidatePlan,
)
