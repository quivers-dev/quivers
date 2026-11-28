"""Log-space belief propagation on explicitly declared finite string supports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import isfinite, log, log1p, prod
from types import MappingProxyType
from typing import Literal

import torch

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import FixedStringQuery, RationalRelation, _weight_snapshot
from quivers.core.relations.graph import StringFactorGraph
from quivers.core.relations.projection import (
    NgramFamily,
    NgramProjection,
    PenalizedNgramProjection,
    _validate_projection_settings,
)
from quivers.core.relations.messages import StringCandidateMessage, _restrict_candidates
from quivers.core.relations.queries import RelationLimitError, _validate_limits

type _Word = tuple[int, ...]
type _Messages = tuple[tuple[torch.Tensor, ...], ...]
type _Occurrence = tuple[int, int]
type _FiniteEPInitialSites = tuple[
    tuple[int, str, tuple[_Word, ...], tuple[float, ...]], ...
]


@dataclass(frozen=True, slots=True)
class StringBeliefResult:
    """The marginal estimates and measured fixed-point residual of a sweep."""

    log_marginals: Mapping[str, torch.Tensor]
    status: Literal["converged", "limit"]
    sweeps: int
    message_residual: float
    log_messages: _Messages


@dataclass(frozen=True, slots=True)
class PrunedStringBeliefResult:
    """Approximate finite beliefs and diagnostics from a beam-limited schedule.

    The log messages use the plan's declared candidate axes. A pruned
    coordinate has log weight negative infinity. The proposal records
    use factor and tape order and report raw masses before restriction
    and renormalization. If an attempted support has no normalized
    beliefs, the result records its messages and the support failure.
    """

    log_marginals: Mapping[str, torch.Tensor] | None
    status: Literal["converged", "limit", "empty_support"]
    sweeps: int
    message_residual: float | None
    log_messages: _Messages
    proposals: tuple[tuple[StringCandidateMessage, ...], ...]
    truncated: bool
    diagnostic: str | None = None
    exact: Literal[False] = False


@dataclass(frozen=True, slots=True)
class StringSiteProjection:
    """The tilted normalization and fit for one factor-to-variable site."""

    factor: int
    variable: str
    log_tilt_mass: torch.Tensor
    fit: NgramProjection | PenalizedNgramProjection | None
    diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class StringEPResult:
    """Finite string beliefs and separate outer and local diagnostics.

    Site projections retain the final pass. The three cumulative work
    counts include all passes and any failing attempt.
    """

    log_marginals: Mapping[str, torch.Tensor]
    status: Literal["converged", "limit", "projection_failed"]
    sweeps: int
    message_residual: float | None
    log_messages: _Messages
    site_projections: tuple[StringSiteProjection, ...]
    projection_attempts: int = 0
    projection_iterations: int = 0
    line_search_trials: int = 0


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


class _EmptyStringSupport(ValueError):
    pass


def _normalized(log_weights: torch.Tensor) -> torch.Tensor:
    normalizer = LOG_PROB.join(log_weights, dim=0)
    value = normalizer.item()
    if value == -torch.inf:
        raise _EmptyStringSupport(
            "a string belief message has no positive finite normalizer"
        )
    if not isfinite(value):
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


def _validate_sweeps(max_sweeps: int, tolerance: float | None, damping: float) -> None:
    if type(max_sweeps) is not int or max_sweeps < 0:
        raise ValueError("max_sweeps must be a nonnegative integer")
    if not ((tolerance is None or 0 <= tolerance < float("inf")) and 0 <= damping < 1):
        raise ValueError("invalid string message tolerance or damping")


def _validate_pruned_settings(
    beam: int, max_sweeps: int, tolerance: float | None, damping: float
) -> None:
    _validate_sweeps(max_sweeps, tolerance, damping)
    if max_sweeps == 0:
        raise ValueError("pruned string beliefs require at least one sweep")
    if type(beam) is not int or beam < 1:
        raise ValueError("the string belief beam must be a positive integer")


def _validate_finite_ep_initial_sites(
    initial_sites: _FiniteEPInitialSites | None,
) -> None:
    if initial_sites is None:
        return
    if not isinstance(initial_sites, tuple):
        raise ValueError("finite EP initial sites require ordered records")
    for record in initial_sites:
        if not isinstance(record, tuple) or len(record) != 4:
            raise ValueError("invalid finite EP initial site record")
        factor, variable, features, coefficients = record
        if (
            type(factor) is not int
            or factor < 0
            or not isinstance(variable, str)
            or not variable
            or not isinstance(features, tuple)
            or any(
                not isinstance(gram, tuple)
                or not gram
                or any(type(symbol) is not int or symbol < 0 for symbol in gram)
                for gram in features
            )
        ):
            raise ValueError("invalid finite EP initial site identity")
        if (
            not isinstance(coefficients, tuple)
            or len(coefficients) != len(features)
            or any(
                type(value) is not float or not isfinite(value)
                for value in coefficients
            )
        ):
            raise ValueError("finite EP initial coefficients must be finite floats")


def _message_residual(old: _Messages, new: _Messages) -> float:
    return max(
        0.5 * (message.exp() - proposed.exp()).abs().sum().item()
        for old_factor, new_factor in zip(old, new, strict=True)
        for message, proposed in zip(old_factor, new_factor, strict=True)
    )


def _damp_on_support(
    old: torch.Tensor, proposal: torch.Tensor, damping: float
) -> torch.Tensor:
    if not damping:
        return proposal
    previous = old.masked_fill(~torch.isfinite(proposal), -torch.inf)
    mass = LOG_PROB.join(previous, dim=0)
    if not bool(torch.isfinite(mass)):
        return proposal
    return LOG_PROB.join(
        torch.stack((previous - mass + log(damping), proposal + log1p(-damping))),
        dim=0,
    )


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
        self._tapes = {
            name: variable.tape for name, variable in graph._variables.items()
        }
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
        self._ep_families: tuple[int, dict[str, NgramFamily]] | None = None

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

    def _start(self) -> tuple[tuple[torch.Tensor, ...], _Messages]:
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
        return tables, messages

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
        _validate_sweeps(max_sweeps, tolerance, damping)
        tables, messages = self._start()
        like = tables[0]
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
            residual = _message_residual(messages, proposed)
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

    def run_pruned(
        self,
        *,
        beam: int,
        max_sweeps: int,
        tolerance: float | None = 1e-9,
        damping: float = 0.0,
    ) -> PrunedStringBeliefResult:
        """Run bounded top-weight messages on the declared finite supports.

        Each raw proposal uses the normalized variable cavities. Its
        full and omitted log masses use this same gauge and are measured
        before normalization after pruning. Ties in the beam are resolved
        lexicographically by word. Damping projects the previous message
        onto the new proposal's positive support and normalizes it before
        mixing. With disjoint support, only the new proposal contributes.

        The residual measures total variation from the undamped pruned
        proposal. Convergence concerns these approximate messages.
        Support failure may occur when hard pruning leaves no consistent
        variable or factor message. At least one sweep updates the
        uniform start before returning a converged or limited result.
        """
        _validate_pruned_settings(beam, max_sweeps, tolerance, damping)
        tables, messages = self._start()
        like = tables[0]
        sweeps = 0
        truncated = False
        records: tuple[tuple[StringCandidateMessage, ...], ...] = ()
        while True:
            try:
                beliefs, cavities = self._variables(messages, like)
                proposals, current = self._pruned_proposals(tables, cavities, beam)
            except _EmptyStringSupport as error:
                return PrunedStringBeliefResult(
                    None,
                    "empty_support",
                    sweeps,
                    None,
                    messages,
                    records,
                    truncated,
                    diagnostic=str(error),
                )
            records = current
            truncated = truncated or any(
                record.status == "truncated" for group in records for record in group
            )
            residual = _message_residual(messages, proposals)
            converged = tolerance is not None and residual <= tolerance
            if sweeps and (converged or sweeps == max_sweeps):
                return PrunedStringBeliefResult(
                    MappingProxyType(beliefs),
                    "converged" if converged else "limit",
                    sweeps,
                    residual,
                    messages,
                    records,
                    truncated,
                )
            messages = tuple(
                tuple(
                    _damp_on_support(old, new, damping)
                    for old, new in zip(old_group, new_group, strict=True)
                )
                for old_group, new_group in zip(messages, proposals, strict=True)
            )
            sweeps += 1

    def _pruned_proposals(
        self,
        tables: tuple[torch.Tensor, ...],
        cavities: dict[_Occurrence, torch.Tensor],
        beam: int,
    ) -> tuple[_Messages, tuple[tuple[StringCandidateMessage, ...], ...]]:
        proposals = []
        records = []
        for factor_id, (values, factor) in enumerate(
            zip(tables, self._tables, strict=True)
        ):
            incoming = tuple(cavities[factor_id, axis] for axis in range(values.ndim))
            projected = []
            local = []
            for name, raw in zip(
                factor.variables, _outgoing(values, incoming), strict=True
            ):
                if bool(torch.isneginf(raw).all()):
                    raise _EmptyStringSupport(
                        f"factor {factor_id} has no positive proposal for {name!r}"
                    )
                selected, record = _restrict_candidates(
                    self._tapes[name],
                    self._supports[name],
                    raw,
                    raw.new_full((), -torch.inf),
                    beam,
                )
                rows = torch.tensor(selected, dtype=torch.long, device=raw.device)
                projected.append(
                    raw.new_full(raw.shape, -torch.inf).index_copy(
                        0, rows, record.normalized_log_weights()
                    )
                )
                local.append(record)
            proposals.append(tuple(projected))
            records.append(tuple(local))
        return tuple(proposals), tuple(records)

    def expectation_propagation(
        self,
        *,
        order: int,
        max_sweeps: int,
        tolerance: float | None = 1e-9,
        damping: float = 0.0,
        projection_tolerance: float = 1e-9,
        projection_penalty: float = 0.0,
        projection_max_iterations: int = 64,
        projection_max_backtracks: int = 16,
        initial_sites: _FiniteEPInitialSites | None = None,
    ) -> StringEPResult:
        """Run fixed n-gram inclusive-KL updates over the declared support.

        A site's projected marginal is divided by its normalized cavity
        and re-expressed as a normalized message. Damping interpolates
        log weights and preserves the family. The outer residual
        measures the undamped message update at the returned state.
        A failed local projection returns the preceding accepted state
        and the separate site diagnostic. A positive projection penalty
        uses the tree-group objective, with a proximal optimality
        residual and the same projection iteration limit. The
        backtracking limit applies to the unpenalized fit. Initial sites
        give factor position, variable, exact affine-basis order and
        finite coefficients. None starts each site at unit weight.
        Autograd follows the executed updates and local accepted steps.
        """
        _validate_sweeps(max_sweeps, tolerance, damping)
        _validate_projection_settings(
            projection_tolerance, projection_max_iterations, projection_max_backtracks
        )
        if not 0 <= projection_penalty < float("inf"):
            raise ValueError("projection_penalty must be nonnegative and finite")
        _validate_finite_ep_initial_sites(initial_sites)
        cached = self._ep_families
        if cached is None or type(order) is not int or order != cached[0]:
            families = {
                name: NgramFamily(words, order)
                for name, words in self._supports.items()
            }
            self._ep_families = order, families
        else:
            _, families = cached
        if initial_sites is not None:
            expected = tuple(
                (f, name, families[name].features)
                for f, factor in enumerate(self._tables)
                for name in factor.variables
            )
            if tuple(record[:3] for record in initial_sites) != expected:
                raise ValueError(
                    "finite EP initial sites must match graph and affine-basis order"
                )
        tables, messages = self._start()
        like = tables[0]
        if like.dtype != torch.float64:
            raise TypeError("projected string messages require float64 factor weights")
        if initial_sites is not None:
            initial = iter(initial_sites)
            messages = tuple(
                tuple(
                    _normalized(
                        like.new_tensor(families[name]._matrix)
                        @ like.new_tensor(next(initial)[3])
                    )
                    for name in factor.variables
                )
                for factor in self._tables
            )
        sweeps = projection_attempts = projection_iterations = line_search_trials = 0
        while True:
            beliefs, cavities = self._variables(messages, like)
            records: list[StringSiteProjection] = []
            proposed: list[tuple[torch.Tensor, ...]] = []
            for f, (table, factor) in enumerate(zip(tables, self._tables, strict=True)):
                local = tuple(cavities[f, axis] for axis in range(table.ndim))
                next_messages: list[torch.Tensor] = []
                for name, incoming, outgoing in zip(
                    factor.variables, local, _outgoing(table, local), strict=True
                ):
                    tilted = incoming + outgoing
                    log_mass = LOG_PROB.join(tilted, dim=0)
                    fit: NgramProjection | PenalizedNgramProjection | None = None
                    projection_attempts += 1
                    if bool(
                        torch.isfinite(log_mass) & ((tilted - log_mass).exp() > 0).all()
                    ):
                        family = families[name]
                        if projection_penalty:
                            fit = family.project_penalized(
                                tilted,
                                penalty=projection_penalty,
                                base_log_weights=incoming,
                                tolerance=projection_tolerance,
                                max_iterations=projection_max_iterations,
                            )
                        else:
                            fit = family.project(
                                tilted,
                                base_log_weights=incoming,
                                tolerance=projection_tolerance,
                                max_iterations=projection_max_iterations,
                                max_backtracks=projection_max_backtracks,
                            )
                    if fit is not None:
                        projection_iterations += fit.iterations
                        if isinstance(fit, NgramProjection):
                            line_search_trials += fit.line_search_trials
                    if fit is None or fit.status != "converged":
                        problem = (
                            "tilted weights must be positive on the full support"
                            if fit is None
                            else f"local projection {fit.status}"
                        )
                        records.append(
                            StringSiteProjection(f, name, log_mass, fit, problem)
                        )
                        return StringEPResult(
                            MappingProxyType(beliefs),
                            "projection_failed",
                            sweeps,
                            None,
                            messages,
                            tuple(records),
                            projection_attempts,
                            projection_iterations,
                            line_search_trials,
                        )
                    records.append(StringSiteProjection(f, name, log_mass, fit))
                    next_messages.append(_normalized(fit.log_probabilities - incoming))
                proposed.append(tuple(next_messages))

            candidates = tuple(proposed)
            residual = _message_residual(messages, candidates)
            converged = tolerance is not None and residual <= tolerance
            if converged or sweeps == max_sweeps:
                return StringEPResult(
                    MappingProxyType(beliefs),
                    "converged" if converged else "limit",
                    sweeps,
                    residual,
                    messages,
                    tuple(records),
                    projection_attempts,
                    projection_iterations,
                    line_search_trials,
                )
            messages = (
                tuple(
                    tuple(
                        _normalized(damping * old + (1 - damping) * new)
                        for old, new in zip(old_factor, new_factor, strict=True)
                    )
                    for old_factor, new_factor in zip(messages, candidates, strict=True)
                )
                if damping
                else candidates
            )
            sweeps += 1


__all__ = [
    "FiniteStringBeliefPlan",
    "StringBeliefResult",
    "PrunedStringBeliefResult",
    "StringEPResult",
    "StringSiteProjection",
]
