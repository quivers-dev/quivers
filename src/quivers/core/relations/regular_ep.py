"""Projected expectation propagation with sparse regular string messages."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch

from quivers.core.algebras import LOG_PROB
from quivers.core.relations import RationalRelation, _weight_snapshot
from quivers.core.relations.beliefs import _validate_sweeps
from quivers.core.relations.graph import StringFactor, StringFactorGraph, _TreeBounds
from quivers.core.relations.ngram_message import NgramMessagePlan
from quivers.core.relations.projection import _validate_projection_settings
from quivers.core.relations.regular_projection import (
    RegularNgramProjection,
    _maximum_length,
    _require_regular_normalizer,
    _snapshot,
    penalized_regular_ngram_projection,
)
from quivers.core.relations.transforms import intersect

type _Word = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RegularStringBelief:
    """A finite regular relation and the log mass used to normalize it."""

    relation: RationalRelation
    log_normalizer: torch.Tensor
    max_states: int
    max_work: int

    def log_prob(self, word: tuple[int, ...]) -> torch.Tensor:
        """Return the fitted belief's normalized log probability of a word."""
        score = self.relation.query(
            (word,), max_states=self.max_states, max_work=self.max_work
        ).weight()
        return score - self.log_normalizer


@dataclass(frozen=True, slots=True)
class RegularStringSite:
    """A scaled n-gram factor-to-variable message."""

    factor: int
    variable: str
    features: tuple[_Word, ...]
    coefficients: torch.Tensor
    log_scale: torch.Tensor


@dataclass(frozen=True, slots=True)
class RegularStringSiteUpdate:
    """One local projection and its tilted mass with normalized cavities."""

    factor: int
    variable: str
    log_tilt_mass: torch.Tensor
    fit: RegularNgramProjection


@dataclass(frozen=True, slots=True)
class RegularStringEPResult:
    """Approximate fitted beliefs, sites, and separate projection diagnostics."""

    marginals: Mapping[str, RegularStringBelief]
    sites: tuple[RegularStringSite, ...]
    status: Literal["converged", "limit", "projection_failed"]
    sweeps: int
    message_residual: float | None
    updates: tuple[RegularStringSiteUpdate, ...]


@dataclass(frozen=True, slots=True)
class _Site:
    coefficients: torch.Tensor
    scale: torch.Tensor

    def damp(self, other: _Site, fraction: float) -> _Site:
        return _Site(
            fraction * self.coefficients + (1 - fraction) * other.coefficients,
            fraction * self.scale + (1 - fraction) * other.scale,
        )


class _ReferenceLanguage:
    def __init__(
        self,
        base: RationalRelation,
        features: tuple[_Word, ...],
        bounds: _TreeBounds,
    ) -> None:
        self.base = base
        self.bounds = bounds
        self.length = _maximum_length(base)
        checked = NgramMessagePlan(
            base, features, max_states=bounds.max_states, max_work=bounds.max_work
        )
        self.features = checked.features
        self.compiled: tuple[tuple[_Word, ...], NgramMessagePlan] | None = None
        self.zeros = base._weights().new_zeros(len(self.features))

    def active(
        self, coefficients: torch.Tensor
    ) -> tuple[tuple[_Word, ...], torch.Tensor]:
        indices = tuple(i for i, value in enumerate(coefficients.tolist()) if value)
        names = tuple(self.features[i] for i in indices)
        gather = torch.tensor(indices, dtype=torch.long, device=coefficients.device)
        return names, coefficients.index_select(0, gather)

    def message(self, coefficients: torch.Tensor) -> RegularStringBelief:
        active, values = self.active(coefficients)
        bounds = self.bounds
        if self.compiled is None or active != self.compiled[0]:
            plan = NgramMessagePlan(
                self.base,
                active,
                max_states=bounds.max_states,
                max_work=bounds.max_work,
            )
            self.compiled = active, plan
        else:
            _, plan = self.compiled
        relation = plan.bind(values)
        mass = relation.total_weight(
            max_states=bounds.max_states, max_work=bounds.max_work
        )
        _require_regular_normalizer(mass, "regular belief or cavity")
        return RegularStringBelief(relation, mass, bounds.max_states, bounds.max_work)


def _projection(
    factor: StringFactor,
    target: int,
    cavities: tuple[RegularStringBelief, ...],
    reference: _ReferenceLanguage,
    *,
    penalty: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[RegularNgramProjection, torch.Tensor, torch.Tensor]:
    bounds = reference.bounds
    likelihood = factor.relation
    outside = cavities[target].log_normalizer.new_zeros(())
    for axis in reversed(range(len(cavities))):
        if axis != target:
            likelihood = bounds.product(likelihood, cavities[axis].relation, axis)
            outside = outside + cavities[axis].log_normalizer
    fit = penalized_regular_ngram_projection(
        cavities[target].relation,
        likelihood,
        reference.features,
        penalty=penalty,
        tolerance=tolerance,
        max_iterations=max_iterations,
        max_states=bounds.max_states,
        max_work=bounds.max_work,
        max_closure_size=bounds.max_closure_size,
    )
    return (
        fit,
        fit.log_site_scale - outside,
        fit.log_tilt_mass - outside - cavities[target].log_normalizer,
    )


def _beliefs(
    references: dict[str, _ReferenceLanguage],
    factors: tuple[StringFactor, ...],
    sites: tuple[tuple[_Site, ...], ...],
) -> tuple[dict[str, RegularStringBelief], dict[tuple[int, int], RegularStringBelief]]:
    copies: dict[str, list[tuple[int, int, _Site]]] = defaultdict(list)
    for index, (factor, row) in enumerate(zip(factors, sites, strict=True)):
        for axis, (name, site) in enumerate(zip(factor.variables, row, strict=True)):
            copies[name].append((index, axis, site))
    marginals = {}
    cavities = {}
    for name, reference in references.items():
        local = copies[name]
        prefixes = [reference.zeros]
        for _, _, site in local:
            prefixes.append(prefixes[-1] + site.coefficients)
        marginals[name] = reference.message(prefixes.pop())
        suffix = reference.zeros
        for prefix, (index, axis, site) in zip(
            reversed(prefixes), reversed(local), strict=True
        ):
            cavities[index, axis] = reference.message(prefix + suffix)
            suffix = suffix + site.coefficients
    return marginals, cavities


@torch.inference_mode(False)
def regular_string_ep(
    graph: StringFactorGraph,
    bases: Mapping[str, RationalRelation],
    features: Mapping[str, tuple[tuple[int, ...], ...]],
    *,
    penalty: float,
    max_sweeps: int,
    max_states: int,
    max_work: int,
    max_closure_size: int | None = None,
    tolerance: float | None = 1e-8,
    damping: float = 0.0,
    projection_tolerance: float = 1e-8,
    projection_max_iterations: int = 512,
) -> RegularStringEPResult:
    """Approximate sum-product over finite regular reference languages.

    Factor scores are multiplied against the supplied base languages.
    A base gives one reference weight per word, with observed variables
    additionally restricted to their observed words. Sites are
    exponentiated n-gram scores with separate log scales. No table
    over all words or assignments is built.

    Parameters
    ----------
    graph : StringFactorGraph
        The typed graph with float64 LogProb factor weights.
    bases : Mapping[str, RationalRelation]
        One acyclic productive reference language per variable.
    features : Mapping[str, tuple[tuple[int, ...], ...]]
        One finite candidate n-gram vocabulary per variable.
    penalty : float
        Nonnegative weight on the local n-gram prefix-group penalty.
    max_sweeps : int
        Maximum synchronous sweeps accepted by the outer solver.
    max_states : int
        Reached state limit for each relation product or query.
    max_work : int
        Work limit for each relation product or query.
    max_closure_size : int or None
        Bound for productive epsilon closure in tilted factors.
    tolerance : float or None
        Upper bound on a site's undamped change in log score,
        including its scale. None uses the full sweep budget.
    damping : float
        Fraction of the previous site retained in log-space damping.
    projection_tolerance : float
        Local proximal gradient tolerance.
    projection_max_iterations : int
        Maximum accepted iterations per local projection.

    Returns
    -------
    RegularStringEPResult
        Approximate marginals at the last accepted sweep. Local
        optimizer failure has a separate status, and relation resource
        errors propagate. Weights are snapshotted and the schedule is
        not differentiated.
    """
    _validate_sweeps(max_sweeps, tolerance, damping)
    _validate_projection_settings(projection_tolerance, projection_max_iterations, 0)
    if not 0 <= penalty < float("inf"):
        raise ValueError("n-gram projection penalty must be nonnegative and finite")
    if set(graph._variables) != set(bases) or set(graph._variables) != set(features):
        raise ValueError(
            "regular string EP requires bases and feature vocabularies for every variable"
        )
    for name, variable in graph._variables.items():
        base = bases[name]
        if base.algebra is not LOG_PROB or base.topology.tapes != (variable.tape,):
            raise TypeError(
                f"regular string EP has a mistyped LogProb base for {name!r}"
            )
    if any(factor.relation.algebra is not LOG_PROB for factor in graph._factors):
        raise TypeError("regular string EP requires LogProb factors")
    bounds = _TreeBounds(max_states, max_work, max_closure_size)
    with _weight_snapshot(), torch.no_grad():
        frozen_bases = {name: _snapshot(base) for name, base in bases.items()}
        factors = tuple(
            StringFactor(factor.variables, _snapshot(factor.relation))
            for factor in graph._factors
        )
    sources = [relation._weights() for relation in frozen_bases.values()]
    sources.extend(factor.relation._weights() for factor in factors)
    if any(
        weights.dtype != torch.float64 or weights.device != sources[0].device
        for weights in sources
    ):
        raise TypeError("regular string EP requires float64 weights on a common device")
    references = {}
    for name, base in frozen_bases.items():
        variable = graph._variables[name]
        if variable.observed is not None:
            base = intersect(
                base,
                bounds.one_word(variable, base),
                max_states=max_states,
                max_work=max_work,
            )
        references[name] = _ReferenceLanguage(base, features[name], bounds)

    sites = tuple(
        tuple(
            _Site(references[name].zeros, references[name].zeros.new_zeros(()))
            for name in factor.variables
        )
        for factor in factors
    )
    sweeps = 0
    while True:
        marginals, cavity = _beliefs(references, factors, sites)
        updates: list[RegularStringSiteUpdate] = []
        report: list[RegularStringSite] = []
        for index, (factor, row) in enumerate(zip(factors, sites, strict=True)):
            for name, site in zip(factor.variables, row, strict=True):
                active, coefficients = references[name].active(site.coefficients)
                report.append(
                    RegularStringSite(index, name, active, coefficients, site.scale)
                )
        proposed: list[tuple[_Site, ...]] = []
        residual = 0.0
        for index, (factor, row) in enumerate(zip(factors, sites, strict=True)):
            incoming = tuple(cavity[index, axis] for axis in range(len(row)))
            next_row: list[_Site] = []
            for axis, (name, current) in enumerate(
                zip(factor.variables, row, strict=True)
            ):
                reference = references[name]
                fit, scale, mass = _projection(
                    factor,
                    axis,
                    incoming,
                    reference,
                    penalty=penalty,
                    tolerance=projection_tolerance,
                    max_iterations=projection_max_iterations,
                )
                updates.append(RegularStringSiteUpdate(index, name, mass, fit))
                if fit.status != "converged":
                    return RegularStringEPResult(
                        MappingProxyType(marginals),
                        tuple(report),
                        "projection_failed",
                        sweeps,
                        None,
                        tuple(updates),
                    )
                candidate = _Site(fit.coefficients, scale)
                difference = (
                    candidate.scale - current.scale
                ).abs() + reference.length * (
                    candidate.coefficients - current.coefficients
                ).abs().sum()
                if not bool(torch.isfinite(difference)):
                    raise OverflowError(
                        "regular string EP residual is outside the evaluation precision"
                    )
                residual = max(residual, difference.item())
                next_row.append(
                    current.damp(candidate, damping) if damping else candidate
                )
            proposed.append(tuple(next_row))
        converged = tolerance is not None and residual <= tolerance
        if converged or sweeps == max_sweeps:
            return RegularStringEPResult(
                MappingProxyType(marginals),
                tuple(report),
                "converged" if converged else "limit",
                sweeps,
                residual,
                tuple(updates),
            )
        sites = tuple(proposed)
        sweeps += 1


__all__ = [
    "RegularStringBelief",
    "RegularStringEPResult",
    "RegularStringSite",
    "RegularStringSiteUpdate",
    "regular_string_ep",
]
