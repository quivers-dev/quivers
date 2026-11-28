"""Compare observation and atomic intervention at one addressed Bernoulli draw."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction

from quivers.qiec._random_site_graph import _Address
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import ExactJvp, JvpUnsupported, _bound
from quivers.qiec.finite_marginalization import (
    _checked_finite_unit,
    _History,
    _positive_joint,
    _prior_histories,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import QiecModule


@dataclass(frozen=True, slots=True)
class AddressedBernoulliEvent:
    """A fixed Boolean outcome at an original canonical sample address.

    Parameters
    ----------
    address : tuple
        The full canonical source request address, including scope frames.
    value : bool
        The Boolean outcome at that address.
    """

    address: _Address
    value: bool

    def __post_init__(self) -> None:
        """Require a Boolean outcome.

        Raises
        ------
        ValueError
            If the supplied outcome is not a Boolean.
        """
        if type(self.value) is not bool:
            raise ValueError("an addressed Bernoulli outcome must be Boolean")


@dataclass(frozen=True, slots=True)
class CausalBernoulliHistory:
    """One history compatible with the selected treatment value.

    Parameters
    ----------
    history : tuple
        The chronological canonical addresses and Boolean outcomes.
    matches_outcome : bool
        Whether this history also has the requested response value.
    observational_weight, interventional_weight : ExactJvp
        The unnormalized weights with the source observation factor.
        The interventional weight omits only the treated kernel.
    """

    history: _History
    matches_outcome: bool
    observational_weight: ExactJvp
    interventional_weight: ExactJvp


@dataclass(frozen=True, slots=True)
class CausalBernoulliComparison:
    """Exact conditional and interventional responses on bounded source.

    Parameters
    ----------
    computation : ComputationId
        The named observational source and its autonomous ordered kernels.
    treatment, outcome : AddressedBernoulliEvent
        The selected treatment and response events.
    conditioned, intervened : ExactJvp
        The normalized response probability and derivative when observing
        or replacing the treated draw, respectively.
    observational_evidence, interventional_evidence : ExactJvp
        Each unnormalized denominator and its derivative.
    histories : tuple[CausalBernoulliHistory, ...]
        The matched source histories and their weights before normalization.
    source_steps : int
        The measured source work across the original finite enumeration.
    """

    computation: ComputationId
    treatment: AddressedBernoulliEvent
    outcome: AddressedBernoulliEvent
    conditioned: ExactJvp
    intervened: ExactJvp
    observational_evidence: ExactJvp
    interventional_evidence: ExactJvp
    histories: tuple[CausalBernoulliHistory, ...]
    source_steps: int


def _quotient(numerator: ExactJvp, denominator: ExactJvp) -> ExactJvp:
    """Divide by the positive dual weight of a kernel or evidence.

    Parameters
    ----------
    numerator, denominator : ExactJvp
        The two exact dual values with a positive denominator.

    Returns
    -------
    ExactJvp
        The quotient and its exact directional derivative.
    """
    value = numerator.value / denominator.value
    tangent = (numerator.tangent - value * denominator.tangent) / denominator.value
    _bound(value, tangent)
    return ExactJvp(value, tangent)


def _sum_weights(items: Iterable[ExactJvp]) -> ExactJvp:
    """Sum the values and derivatives of a finite collection of weights.

    Parameters
    ----------
    items : Iterable[ExactJvp]
        Exact atom weights and their derivatives.

    Returns
    -------
    ExactJvp
        The finite exact sum, including the empty sum at zero.
    """
    value = tangent = Fraction(0)
    for item in items:
        value += item.value
        tangent += item.tangent
        _bound(value, tangent)
    return ExactJvp(value, tangent)


def compare_bernoulli_do_and_condition(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    treatment: AddressedBernoulliEvent,
    outcome: AddressedBernoulliEvent,
    max_paths: int = 4096,
    fuel: int = 50_000,
) -> CausalBernoulliComparison:
    """Compare observing and fixing the same source Bernoulli outcome.

    The checked scalar source specifies the joint process and returns one
    positive observation factor per history. Both queried addresses must
    occur in every history. The comparison treats the individual source
    kernels as autonomous mechanisms of this specified model.

    Observational conditioning retains the probability of the chosen
    treatment in the original joint distribution. The intervention
    replaces that one conditional kernel by the chosen constant. Both
    sides retain the source observation factors, and normalize with their
    own evidence and quotient rule. Earlier and later kernels retain
    their original probabilities at their actual histories.

    Parameters
    ----------
    module : QiecModule
        The source supplying the actual Bernoulli kernels.
    computation : ComputationId
        The scalar entry on one nominal Random instance.
    arguments, direction : tuple[int or Fraction, ...]
        The parameter point and exact directional perturbation.
    policies : tuple[SiteEstimator, ...]
        An explicit enumerate policy for each reached site.
    treatment, outcome : AddressedBernoulliEvent
        The fixed treatment and the queried response value.
    max_paths, fuel : int
        Bounds on complete histories and steps per reference replay.

    Returns
    -------
    CausalBernoulliComparison
        Exact responses, evidence, and the contributing addressed histories.

    Raises
    ------
    JvpUnsupported
        If an address is missing on a history, an observation factor is
        nonpositive, or the source leaves the finite Bernoulli fragment.
    JvpBoundaryError
        If a parameter-dependent decision meets a comparison boundary.
    JvpLimit
        If the arithmetic or reference exploration exceeds a bound.
    ValueError
        If the entry, arguments or limits are malformed.
    """
    finite = _checked_finite_unit(
        module, computation, arguments, direction, policies, max_paths, fuel
    )
    visits = {visit.prefix: visit for visit in finite.visits}
    histories: list[CausalBernoulliHistory] = []
    action_priors: list[ExactJvp] = []
    for history, prior, factor in _prior_histories(finite.tree):
        prefix = tuple(choice for _, choice in history)
        addressed = {address: index for index, (address, _) in enumerate(history)}
        target_index = addressed.get(treatment.address)
        response_index = addressed.get(outcome.address)
        if target_index is None or response_index is None:
            missing = "treatment" if target_index is None else "outcome"
            raise JvpUnsupported(
                f"the {missing} address must occur on every source history, missing at {prefix!r}"
            )
        observed = _positive_joint(prior, factor, prefix)
        if prefix[target_index] != treatment.value:
            continue
        visit = visits[prefix[:target_index]]
        assert visit.address == treatment.address
        kernel = (
            ExactJvp(visit.probability, visit.tangent)
            if treatment.value
            else ExactJvp(1 - visit.probability, -visit.tangent)
        )
        action_priors.append(_quotient(prior, kernel))
        histories.append(
            CausalBernoulliHistory(
                history,
                prefix[response_index] == outcome.value,
                observed,
                _quotient(observed, kernel),
            )
        )
    assert _sum_weights(action_priors) == ExactJvp(Fraction(1), Fraction(0))
    obs_evidence = _sum_weights(atom.observational_weight for atom in histories)
    do_evidence = _sum_weights(atom.interventional_weight for atom in histories)
    selected = tuple(atom for atom in histories if atom.matches_outcome)
    obs_response = _sum_weights(atom.observational_weight for atom in selected)
    do_response = _sum_weights(atom.interventional_weight for atom in selected)
    return CausalBernoulliComparison(
        computation,
        treatment,
        outcome,
        _quotient(obs_response, obs_evidence),
        _quotient(do_response, do_evidence),
        obs_evidence,
        do_evidence,
        tuple(histories),
        finite.source_steps,
    )


__all__ = [
    "AddressedBernoulliEvent",
    "CausalBernoulliComparison",
    "CausalBernoulliHistory",
    "compare_bernoulli_do_and_condition",
]
