"""Exact finite marginalization and checked complete-data scores."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from quivers.qiec.bernoulli_estimators import (
    BernoulliBranch,
    plan_bernoulli_gradient,
)
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import ExactJvp, JvpUnsupported, _bound
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import QiecModule


@dataclass(frozen=True, slots=True)
class FisherOutcome:
    """One complete finite history with its prior and posterior actions.

    Parameters
    ----------
    prefix : tuple[bool, ...]
        The chronological Bernoulli outcomes for this complete history.
    prior : ExactJvp
        The prior path probability and its directional derivative.
    observation : ExactJvp
        The positive observation factor returned by this source path.
    joint : ExactJvp
        The joint weight, given by the prior times the observation factor.
    posterior : ExactJvp
        The normalized probability of this path and its derivative.
    complete_score : Fraction
        The directional derivative of the log joint weight.
    """

    prefix: tuple[bool, ...]
    prior: ExactJvp
    observation: ExactJvp
    joint: ExactJvp
    posterior: ExactJvp
    complete_score: Fraction


@dataclass(frozen=True, slots=True)
class FisherMarginalization:
    """The exact marginalized unit and its Fisher score.

    Parameters
    ----------
    computation : ComputationId
        The checked source declaring the finite latent model.
    normalizer : ExactJvp
        The marginal observation weight and its derivative.
    marginal_score : Fraction
        The directional derivative of the log normalizer, equal to
        the posterior expectation of the complete-data score.
    outcomes : tuple[FisherOutcome, ...]
        The full finite histories and their exact posterior actions.
    source_steps : int
        The measured source work over all checked reference replays.
    replays : int
        The number of complete and choice-suspended source runs.
    """

    computation: ComputationId
    normalizer: ExactJvp
    marginal_score: Fraction
    outcomes: tuple[FisherOutcome, ...]
    source_steps: int
    replays: int


def marginalize_bernoulli_unit(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    max_paths: int = 4096,
    fuel: int = 50_000,
) -> FisherMarginalization:
    """Marginalize a source unit over its finite Bernoulli alternatives.

    The selected scalar computation returns the observation factor for
    each complete history. Every factor must be strictly positive at the
    given point. The checked Bernoulli and scalar rules ensure stable
    branches, open support and locally differentiable probabilities and
    factors. Positivity and the path bound give a valid finite posterior
    and log normalizer on a neighborhood of that point.

    The marginalized score is exact without evaluating a logarithm. The
    function compares the posterior-weighted complete-data scores with
    the ordinary marginal derivative divided by the marginal weight.

    Parameters
    ----------
    module : QiecModule
        The source declaring the latent choices and observation factors.
    computation : ComputationId
        The checked scalar entry on one nominal Random row.
    arguments, direction : tuple[int or Fraction, ...]
        The exact parameter point and direction in parameter order.
    policies : tuple[SiteEstimator, ...]
        An explicit enumeration declaration for each reached static site.
    max_paths : int
        Maximum number of complete latent histories.
    fuel : int
        Maximum reference steps for each source run.

    Returns
    -------
    FisherMarginalization
        The evidence, posterior branch probabilities and directional scores.

    Raises
    ------
    JvpUnsupported
        If a policy excludes enumeration, a factor is nonpositive, or the
        checked probabilistic source leaves the finite scalar fragment.
    JvpBoundaryError
        If a parameter-dependent decision meets an equality boundary.
    JvpLimit
        If an arithmetic, path or replay bound is exceeded.
    ValueError
        If a bound, computation identity or input vector is malformed.
    """
    if any(policy.estimator != "enumerate" for policy in policies):
        raise JvpUnsupported(
            "finite marginalization requires enumeration at every site"
        )
    finite = plan_bernoulli_gradient(
        module,
        computation,
        arguments,
        direction,
        policies,
        max_paths=max_paths,
        fuel=fuel,
    )
    normalizer = finite.exact
    pending: list[tuple[tuple[bool, ...], ExactJvp, ExactJvp | BernoulliBranch]] = [
        ((), ExactJvp(Fraction(1), Fraction(0)), finite.tree)
    ]
    positive: list[tuple[tuple[bool, ...], ExactJvp, ExactJvp, ExactJvp]] = []
    while pending:
        prefix, prior, node = pending.pop()
        if isinstance(node, BernoulliBranch):
            p, dp = node.visit.probability, node.visit.tangent
            for choice, mass, tangent, child in (
                (True, p, dp, node.yes),
                (False, 1 - p, -dp, node.no),
            ):
                next_prior = ExactJvp(
                    prior.value * mass,
                    prior.tangent * mass + prior.value * tangent,
                )
                _bound(next_prior.value, next_prior.tangent)
                pending.append(((*prefix, choice), next_prior, child))
        else:
            if node.value <= 0:
                raise JvpUnsupported(
                    f"finite Fisher scores need positive observation factors, "
                    f"but history {prefix!r} has a nonpositive factor"
                )
            joint = ExactJvp(
                prior.value * node.value,
                prior.tangent * node.value + prior.value * node.tangent,
            )
            _bound(joint.value, joint.tangent)
            positive.append((prefix, prior, node, joint))

    assert normalizer.value > 0
    marginal_score = normalizer.tangent / normalizer.value
    _bound(marginal_score)
    outcomes: list[FisherOutcome] = []
    posterior_score = Fraction(0)
    for prefix, prior, observation, joint in positive:
        complete_score = joint.tangent / joint.value
        mass = joint.value / normalizer.value
        posterior = ExactJvp(mass, mass * (complete_score - marginal_score))
        _bound(complete_score, posterior.value, posterior.tangent)
        posterior_score += posterior.value * complete_score
        _bound(posterior_score)
        outcomes.append(
            FisherOutcome(prefix, prior, observation, joint, posterior, complete_score)
        )
    assert posterior_score == marginal_score
    return FisherMarginalization(
        computation,
        normalizer,
        marginal_score,
        tuple(outcomes),
        finite.source_steps,
        finite.replays,
    )


__all__ = ["FisherMarginalization", "FisherOutcome", "marginalize_bernoulli_unit"]
