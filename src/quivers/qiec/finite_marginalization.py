"""Exact finite marginalization and checked complete-data scores."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from quivers.qiec._random_site_graph import _Address
from quivers.qiec.bernoulli_estimators import (
    BernoulliBranch,
    BernoulliGradientPlan,
    plan_bernoulli_gradient,
)
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import ExactJvp, JvpUnsupported, _bound
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import QiecModule

type _History = tuple[tuple[_Address, bool], ...]


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


def _checked_finite_unit(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    max_paths: int,
    fuel: int,
) -> BernoulliGradientPlan:
    """Build an exact, exhaustively checked unit from nominal source.

    Parameters
    ----------
    module : QiecModule
        The source module.
    computation : ComputationId
        Its selected scalar entry.
    arguments, direction : tuple
        The parameter point and directional perturbation in positional order.
    policies : tuple[SiteEstimator, ...]
        The explicit enumeration policies.
    max_paths, fuel : int
        The bounds on complete histories and steps per replay.

    Returns
    -------
    BernoulliGradientPlan
        The exact finite source tree and its raw expectation.

    Raises
    ------
    JvpUnsupported
        If the source or the declared estimators leave the exact fragment.
    """
    if any(policy.estimator != "enumerate" for policy in policies):
        raise JvpUnsupported("finite inference requires enumeration at every site")
    return plan_bernoulli_gradient(
        module,
        computation,
        arguments,
        direction,
        policies,
        max_paths=max_paths,
        fuel=fuel,
    )


def _prior_histories(
    tree: ExactJvp | BernoulliBranch,
) -> Iterator[tuple[_History, ExactJvp, ExactJvp]]:
    """Read addressed histories and prior masses from the original tree.

    Parameters
    ----------
    tree : ExactJvp or BernoulliBranch
        The complete checked Bernoulli tree.

    Yields
    ------
    tuple
        The chronological address-outcome history, prior JVP and return JVP.
    """
    pending: list[tuple[_History, ExactJvp, ExactJvp | BernoulliBranch]] = [
        ((), ExactJvp(Fraction(1), Fraction(0)), tree)
    ]
    while pending:
        history, prior, node = pending.pop()
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
                pending.append(
                    ((*history, (node.visit.address, choice)), next_prior, child)
                )
        else:
            yield history, prior, node


def _positive_joint(
    prior: ExactJvp, node: ExactJvp, prefix: tuple[bool, ...]
) -> ExactJvp:
    """Combine a prior with a strictly positive source observation factor.

    Parameters
    ----------
    prior, node : ExactJvp
        The source prior mass and returned factor, with their derivatives.
    prefix : tuple[bool, ...]
        The complete source choice history for refusal diagnostics.

    Returns
    -------
    ExactJvp
        The joint atom with its full derivative.

    Raises
    ------
    JvpUnsupported
        If a complete history has no positive observation factor.
    """
    if node.value <= 0:
        raise JvpUnsupported(
            f"finite inference needs positive observation factors, "
            f"but history {prefix!r} has a nonpositive factor"
        )
    joint = ExactJvp(
        prior.value * node.value,
        prior.tangent * node.value + prior.value * node.tangent,
    )
    _bound(joint.value, joint.tangent)
    return joint


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
    finite = _checked_finite_unit(
        module, computation, arguments, direction, policies, max_paths, fuel
    )
    normalizer = finite.exact
    positive: list[tuple[tuple[bool, ...], ExactJvp, ExactJvp, ExactJvp]] = []
    for history, prior, node in _prior_histories(finite.tree):
        prefix = tuple(choice for _, choice in history)
        positive.append((prefix, prior, node, _positive_joint(prior, node, prefix)))

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
