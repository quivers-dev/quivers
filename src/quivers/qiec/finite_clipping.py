"""Exact pointwise bias of clipping a finite checked gradient estimator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction

from quivers.qiec.bernoulli_estimators import (
    BernoulliGradientPlan,
    BernoulliGradientSample,
    BernoulliVisit,
    _interpret,
    plan_bernoulli_gradient,
)
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import ExactJvp, JvpLimit, _bound, _fraction
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import QiecModule


@dataclass(frozen=True, slots=True)
class ClippedEstimatorDecision:
    """One actual random choice during the finite estimator evaluation.

    Parameters
    ----------
    visit : BernoulliVisit
        The original source occurrence, policy and probability used.
    choice : bool
        The score draw or weak selector at this occurrence.
    """

    visit: BernoulliVisit
    choice: bool


@dataclass(frozen=True, slots=True)
class ClippedEstimatorOutcome:
    """The raw and clipped actions for one complete estimator script.

    Parameters
    ----------
    decisions : tuple[ClippedEstimatorDecision, ...]
        The actual stochastic choices in evaluation order.
    probability : Fraction
        The probability of this entire estimator script.
    raw, clipped : ExactJvp
        The original estimate and the estimate with its tangent clipped.
    """

    decisions: tuple[ClippedEstimatorDecision, ...]
    probability: Fraction
    raw: ExactJvp
    clipped: ExactJvp


@dataclass(frozen=True, slots=True)
class FiniteClippedEstimatorAudit:
    """A pointwise clipping audit bound to one checked estimator execution.

    Parameters
    ----------
    module : QiecModule
        The actual source used to construct the checked estimator.
    source : BernoulliGradientPlan
        The checked objective, occurrence identities and policy tree.
    point, direction : tuple[Fraction, ...]
        The supplied parameter point and direction in source order.
    clip_limit : Fraction
        The symmetric cap applied to each realized directional estimate.
    mean : ExactJvp
        The exact mean of the clipped estimator at this point and direction.
    signed_bias : Fraction
        The mean clipped derivative minus the exact target derivative.
    outcomes : tuple[ClippedEstimatorOutcome, ...]
        The complete finite law of the estimator's own random scripts.
    """

    module: QiecModule = field(repr=False)
    source: BernoulliGradientPlan = field(repr=False)
    point: tuple[Fraction, ...]
    direction: tuple[Fraction, ...]
    clip_limit: Fraction
    mean: ExactJvp
    signed_bias: Fraction
    outcomes: tuple[ClippedEstimatorOutcome, ...]

    def sample(self, *, seed: int | None = None) -> BernoulliGradientSample:
        """Draw a clipped action with the original estimator's choices.

        Parameters
        ----------
        seed : int or None
            An optional seed for the original exact finite estimator.

        Returns
        -------
        BernoulliGradientSample
            The sample and its unchanged trace of choices, with its final
            tangent capped at the audit limit.
        """
        estimate = self.source.sample(seed=seed)
        return replace(estimate, tangent=_clip(estimate.tangent, self.clip_limit))


class _NeedEstimatorChoice(Exception):
    """The next actual estimator choice lies beyond the supplied script.

    Parameters
    ----------
    visit : BernoulliVisit
        The stochastic estimator occurrence that needs its own Boolean.
    """

    def __init__(self, visit: BernoulliVisit) -> None:
        super().__init__(visit.address)
        self.visit = visit


def _clip(tangent: Fraction, limit: Fraction) -> Fraction:
    """Clip one final directional estimate at a symmetric bound.

    Parameters
    ----------
    tangent : Fraction
        The raw finite derivative estimate.
    limit : Fraction
        The nonnegative clipping magnitude.

    Returns
    -------
    Fraction
        The tangent projected onto the closed symmetric interval.
    """
    return max(-limit, min(limit, tangent))


def audit_clipped_bernoulli_gradient(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    clip_limit: int | Fraction,
    max_outcomes: int = 4096,
    max_paths: int = 4096,
    fuel: int = 50_000,
) -> FiniteClippedEstimatorAudit:
    """Compute the signed bias of clipping the final finite directional action.

    The base estimator uses the source's explicit enumerate, score and
    measure policies. This audit drives the same interpreter with a
    complete enumeration of its own score draws and weak selectors.
    It takes the mean after clipping each realized tangent. The exact
    unclipped expectation agrees with the source objective.

    The result attests to the supplied source, point, direction, policies
    and clipping operation. It describes a finite numerical calculation
    at that point.

    Parameters
    ----------
    module : QiecModule
        The actual checked source of the finite Bernoulli estimator.
    computation : ComputationId
        The named scalar entry on one nominal Random instance.
    arguments, direction : tuple[int or Fraction, ...]
        The exact parameter point and directional seed.
    policies : tuple[SiteEstimator, ...]
        The explicit source estimator at each reached static site.
    clip_limit : int or Fraction
        The nonnegative symmetric cap on the final sampled tangent.
    max_outcomes : int
        Maximum number of complete stochastic estimator scripts.
    max_paths, fuel : int
        The independent source path limit and per-replay step limit.

    Returns
    -------
    FiniteClippedEstimatorAudit
        The checked source estimator, signed clipping bias and exact law.

    Raises
    ------
    ValueError
        If the clipping limit, bounds or exact inputs are malformed.
    JvpUnsupported
        If the checked source or its policies leave the finite fragment.
    JvpBoundaryError
        If an active input comparison hits an equality boundary.
    JvpLimit
        If the source, exact arithmetic or script enumeration is exhausted.
    """
    _validate_limit("max_outcomes", max_outcomes)
    cap = _fraction(clip_limit)
    if cap < 0:
        raise ValueError("the finite clipping limit must be nonnegative")
    _bound(cap)
    point = tuple(map(_fraction, arguments))
    tangent = tuple(map(_fraction, direction))
    source = plan_bernoulli_gradient(
        module,
        computation,
        point,
        tangent,
        policies,
        max_paths=max_paths,
        fuel=fuel,
    )
    pending: list[tuple[tuple[bool, ...], Fraction]] = [((), Fraction(1))]
    outcomes: list[ClippedEstimatorOutcome] = []
    count = 1
    mass = value = raw_tangent = clipped_tangent = Fraction(0)
    while pending:
        script, probability = pending.pop()
        decisions: list[ClippedEstimatorDecision] = []

        def choose(visit: BernoulliVisit) -> bool:
            """Supply one scripted choice and retain its actual provenance.

            Parameters
            ----------
            visit : BernoulliVisit
                The reached stochastic policy occurrence.

            Returns
            -------
            bool
                The next scripted Boolean value.

            Raises
            ------
            _NeedEstimatorChoice
                If the reached occurrence extends the supplied script.
            """
            position = len(decisions)
            if position == len(script):
                raise _NeedEstimatorChoice(visit)
            choice = script[position]
            decisions.append(ClippedEstimatorDecision(visit, choice))
            return choice

        try:
            estimate = _interpret(source.tree, choose)
        except _NeedEstimatorChoice as choice:
            count += 1
            if count > max_outcomes:
                raise JvpLimit(
                    f"clipping audit exceeds {max_outcomes} complete estimator scripts"
                ) from None
            p = choice.visit.probability
            yes, no = probability * p, probability * (1 - p)
            _bound(yes, no)
            pending.extend((((*script, True), yes), ((*script, False), no)))
        else:
            assert len(decisions) == len(script)
            raw = ExactJvp(estimate.value, estimate.tangent)
            capped = ExactJvp(estimate.value, _clip(estimate.tangent, cap))
            outcomes.append(
                ClippedEstimatorOutcome(tuple(decisions), probability, raw, capped)
            )
            mass += probability
            value += probability * raw.value
            raw_tangent += probability * raw.tangent
            clipped_tangent += probability * capped.tangent
            _bound(mass, value, raw_tangent, clipped_tangent)
    assert mass == 1 and ExactJvp(value, raw_tangent) == source.exact
    bias = clipped_tangent - source.exact.tangent
    _bound(bias)
    return FiniteClippedEstimatorAudit(
        module,
        source,
        point,
        tangent,
        cap,
        ExactJvp(value, clipped_tangent),
        bias,
        tuple(outcomes),
    )


__all__ = [
    "ClippedEstimatorDecision",
    "ClippedEstimatorOutcome",
    "FiniteClippedEstimatorAudit",
    "audit_clipped_bernoulli_gradient",
]
