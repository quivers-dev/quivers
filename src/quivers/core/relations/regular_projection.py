"""Inclusive group-penalized projection of sparse regular language weights."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Literal

import torch
from torch import nn

from quivers.core.algebras import LOG_PROB
from quivers.core.relations import RationalRelation, _weight_snapshot
from quivers.core.relations.ngram_message import NgramMessagePlan
from quivers.core.relations.paths import _require_normalizer
from quivers.core.relations.projection import (
    _prefix_indices,
    _prefix_prox,
    _validate_projection_settings,
)
from quivers.core.relations.queries import _validate_limits
from quivers.core.relations.transforms import intersect, trim

type _Feature = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RegularNgramProjection:
    """An inclusive projection and the compact message for its active features.

    The full coefficients and both moment vectors follow candidate
    feature order. The message uses a frozen weight snapshot and has
    unnormalized mass exp(log_normalizer). Its density relative to the
    original base is the exponential of the fitted feature score.
    The objective change is relative to the unpenalized base at zero.
    """

    status: Literal["converged", "limit", "stalled"]
    iterations: int
    coefficients: torch.Tensor
    active_features: tuple[_Feature, ...]
    target_moments: torch.Tensor
    fitted_moments: torch.Tensor
    log_base_mass: torch.Tensor
    log_tilt_mass: torch.Tensor
    log_normalizer: torch.Tensor
    objective_change: torch.Tensor
    regularization: torch.Tensor
    optimality_residual: float
    message: RationalRelation
    full_candidate_states: int
    retained_states: int

    @property
    def log_site_scale(self) -> torch.Tensor:
        """Log scale making the fitted tilt's total mass equal the target's."""
        return self.log_tilt_mass - self.log_normalizer


class _MomentPlan:
    def __init__(
        self,
        plan: NgramMessagePlan,
        anchor: torch.Tensor,
        max_states: int,
        max_work: int,
        max_closure_size: int | None,
    ) -> None:
        self._probe = nn.Parameter(anchor.new_zeros(len(plan.features)))
        self._query = plan.bind(self._probe)._total_query(
            max_states, max_work, max_closure_size
        )

    def __call__(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self._probe.copy_(coefficients)
        with torch.enable_grad(), _weight_snapshot():
            normalizer = self._query.weight()
            _require_normalizer(normalizer, LOG_PROB)
            if len(self._probe) and normalizer.requires_grad:
                (moment,) = torch.autograd.grad(
                    normalizer, self._probe, allow_unused=True
                )
            else:
                moment = None
        return normalizer.detach(), torch.zeros_like(
            self._probe
        ) if moment is None else moment.detach()


def _maximum_length(base: RationalRelation) -> int:
    topology = trim(base).topology
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for arc in topology.arcs:
        children[arc.source].append((arc.target, int(arc.labels[0] is not None)))
    for mark in topology.final:
        children.setdefault(mark.state, [])
    dependencies = {
        state: [target for target, _ in edges] for state, edges in children.items()
    }
    try:
        order = tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as error:
        raise ValueError(
            "regular n-gram projection requires acyclic productive base support"
        ) from error
    suffix: dict[int, int] = {}
    for state in order:
        suffix[state] = max(
            (suffix[target] + increment for target, increment in children[state]),
            default=0,
        )
    return max((suffix[mark.state] for mark in topology.initial), default=0)


def _snapshot(relation: RationalRelation) -> RationalRelation:
    weights = relation._weights()
    return RationalRelation(
        relation.topology, weights.detach().clone(), algebra=LOG_PROB
    )


@torch.inference_mode(False)
def penalized_regular_ngram_projection(
    base: RationalRelation,
    likelihood: RationalRelation,
    features: tuple[tuple[int, ...], ...],
    *,
    penalty: float,
    tolerance: float = 1e-9,
    max_iterations: int = 512,
    max_states: int,
    max_work: int,
    max_closure_size: int | None = None,
) -> RegularNgramProjection:
    """Fit an active n-gram family to a regular tilted distribution.

    The target multiplies the base by the supplied likelihood. The
    fitted exponential tilt minimizes forward KL from that target plus
    a sum of prefix-group coefficient norms. Complete string tables
    are unnecessary: moments are derivatives of sparse log path sums.

    Parameters
    ----------
    base : RationalRelation
        One float64 LogProb language with acyclic productive support.
    likelihood : RationalRelation
        One float64 LogProb language on the same tape. The product
        with the base must have positive finite mass.
    features : tuple[tuple[int, ...], ...]
        Candidate feature vocabulary in coefficient order.
    penalty : float
        Nonnegative finite coefficient on the sum of prefix norms.
        Zero uses the unpenalized forward-KL objective.
    tolerance : float
        Tolerance on the proximal gradient mapping's infinity norm.
    max_iterations : int
        Upper bound on accepted proximal gradient steps.
    max_states : int
        Reached state limit for each sparse relation and query.
    max_work : int
        Work limit for each sparse relation and query.
    max_closure_size : int or None
        Explicit component limit if the tilted likelihood has
        productive epsilon cycles.

    Returns
    -------
    RegularNgramProjection
        The optimizer status and current fit. The returned relation
        includes only active features. Inputs are read once and
        snapshotted and the optimizer is not differentiated.
    """
    if base.algebra is not LOG_PROB or likelihood.algebra is not LOG_PROB:
        raise TypeError("regular n-gram projection requires LogProb relations")
    if (
        len(base.topology.tapes) != 1
        or base.topology.tapes != likelihood.topology.tapes
    ):
        raise TypeError(
            "regular n-gram projection requires matching one-tape languages"
        )
    if max_states is None or max_work is None:
        raise ValueError(
            "regular n-gram projection requires explicit state and work limits"
        )
    _validate_limits(max_states, max_work, max_closure_size)
    _validate_projection_settings(tolerance, max_iterations, 0)
    if not 0 <= penalty < float("inf"):
        raise ValueError("n-gram projection penalty must be nonnegative and finite")
    length = _maximum_length(base)
    with _weight_snapshot(), torch.no_grad():
        base = _snapshot(base)
        likelihood = _snapshot(likelihood)
    anchor = base._weights()
    other = likelihood._weights()
    if (
        anchor.dtype != torch.float64
        or other.dtype != anchor.dtype
        or other.device != anchor.device
    ):
        raise TypeError(
            "regular n-gram projection requires float64 weights on a common device"
        )

    base_plan = NgramMessagePlan(
        base, features, max_states=max_states, max_work=max_work
    )
    features = base_plan.features
    tilted = intersect(base, likelihood, max_states=max_states, max_work=max_work)
    tilt_plan = NgramMessagePlan(
        tilted, features, max_states=max_states, max_work=max_work
    )
    evaluate_base = _MomentPlan(base_plan, anchor, max_states, max_work, None)
    evaluate_tilt = _MomentPlan(
        tilt_plan, anchor, max_states, max_work, max_closure_size
    )
    coefficients = anchor.new_zeros(len(features))
    tilt_mass, target = evaluate_tilt(coefficients)
    base_mass, fitted = evaluate_base(coefficients)
    normalizer = base_mass
    # Every count is bounded by the maximum accepted length. Hence
    # covariance in any direction is bounded by |features| * length**2.
    radius = max(1, len(features) * length**2)
    step = 1 / radius
    if (
        step == 0
        or not step * penalty < float("inf")
        or (penalty and step * penalty == 0)
    ):
        raise OverflowError(
            "regular n-gram projection step exceeds the evaluation precision"
        )
    groups = tuple(
        torch.tensor(group, dtype=torch.long, device=anchor.device)
        for group in _prefix_indices(features)
    )
    status: Literal["converged", "limit", "stalled"] = "limit"
    iteration = 0
    residual = float("inf")
    for iteration in range(max_iterations + 1):
        gradient = fitted - target
        candidate = coefficients - step * gradient
        if penalty:
            candidate = _prefix_prox(candidate, groups, step * penalty)
        mapping = (coefficients - candidate) / step
        residual = mapping.abs().max().item() if mapping.numel() else 0.0
        if not residual < float("inf"):
            status = "stalled"
            break
        if residual <= tolerance:
            status = "converged"
            break
        if iteration == max_iterations:
            break
        coefficients = candidate
        normalizer, fitted = evaluate_base(coefficients)

    norms = sum(
        (
            torch.linalg.vector_norm(coefficients.index_select(0, group))
            for group in groups
        ),
        anchor.new_zeros(()),
    )
    regularization = penalty * norms
    objective = normalizer - base_mass - coefficients @ target + regularization
    active = tuple(i for i, value in enumerate(coefficients.tolist()) if value)
    retained = tuple(features[i] for i in active)
    compact = NgramMessagePlan(base, retained, max_states=max_states, max_work=max_work)
    active_ids = torch.tensor(active, dtype=torch.long, device=anchor.device)
    message = compact.bind(coefficients.index_select(0, active_ids))
    return RegularNgramProjection(
        status,
        iteration,
        coefficients,
        retained,
        target,
        fitted,
        base_mass,
        tilt_mass,
        normalizer,
        objective,
        regularization,
        residual,
        message,
        base_plan.n_states,
        compact.n_states,
    )


__all__ = ["RegularNgramProjection", "penalized_regular_ngram_projection"]
