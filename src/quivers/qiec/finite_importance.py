"""Exact importance estimates over checked finite addressed source histories."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from fractions import Fraction
from random import Random

from quivers.qiec._random_site_graph import _Address
from quivers.qiec.bernoulli_estimators import BernoulliBranch
from quivers.qiec.contracts import SiteEstimator
from quivers.qiec.differentiation import ExactJvp, JvpUnsupported, _bound, _fraction
from quivers.qiec.finite_marginalization import (
    _checked_finite_unit,
    _History,
    _positive_joint,
    _prior_histories,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.module import QiecModule


class ImportanceCoverageError(JvpUnsupported):
    """A positive target history has no proposal probability.

    Parameters
    ----------
    history : tuple
        The original chronological addresses and Boolean outcomes.
    target_weight : Fraction
        The positive target joint weight of this history.
    """

    def __init__(self, history: _History, target_weight: Fraction) -> None:
        super().__init__(
            f"finite proposal does not cover positive target history {history!r}"
        )
        self.history = history
        self.target_weight = target_weight


@dataclass(frozen=True, slots=True)
class ImportanceCoverage:
    """Evidence that the checked proposal covers the target's addressed support.

    Parameters
    ----------
    computation : ComputationId
        The same nominal source computation used at both points.
    target_point, proposal_point : tuple[Fraction, ...]
        The exact parameter points whose supports were compared.
    histories : tuple
        Every positive target history, all matched by the proposal.
        Histories retain both the original addresses and the Boolean outcomes.
    """

    computation: ComputationId
    target_point: tuple[Fraction, ...]
    proposal_point: tuple[Fraction, ...]
    histories: tuple[_History, ...]


@dataclass(frozen=True, slots=True)
class ImportanceOutcome:
    """One proposal outcome and its exact held-fixed importance action.

    Parameters
    ----------
    history : tuple
        The original chronological addresses and Boolean outcomes.
    proposal : Fraction
        The ordinary prior probability at the distinct proposal point.
    target : ExactJvp
        The target joint weight and its directional derivative.
    estimate : ExactJvp
        The joint weight and derivative each divided by the same held-fixed
        proposal probability.
    """

    history: _History
    proposal: Fraction
    target: ExactJvp
    estimate: ExactJvp


@dataclass(frozen=True, slots=True)
class FiniteImportancePlan:
    """Exact coverage, expectation and proposal actions on the finite source.

    Parameters
    ----------
    coverage : ImportanceCoverage
        The support certified against the original addressed histories.
    exact : ExactJvp
        The unnormalized evidence and its directional derivative.
    outcomes : tuple[ImportanceOutcome, ...]
        The exact proposal law and estimator for each complete history.
    target_steps, proposal_steps : int
        The separately measured reference work at the two points.
    proposal_tree : ExactJvp or BernoulliBranch
        The checked proposal source tree used to make the finite choices.
    """

    coverage: ImportanceCoverage
    exact: ExactJvp
    outcomes: tuple[ImportanceOutcome, ...]
    target_steps: int
    proposal_steps: int
    proposal_tree: ExactJvp | BernoulliBranch = field(repr=False)

    def sample(self, *, seed: int | None = None) -> ImportanceOutcome:
        """Draw one history from the original proposal probabilities.

        Parameters
        ----------
        seed : int or None
            An optional seed for the proposal's independent integer draws.

        Returns
        -------
        ImportanceOutcome
            The chosen history and one exact unnormalized importance estimate.
        """
        random = Random(seed)
        node = self.proposal_tree
        history: list[tuple[_Address, bool]] = []
        while isinstance(node, BernoulliBranch):
            p = node.visit.probability
            choice = random.randrange(p.denominator) < p.numerator
            history.append((node.visit.address, choice))
            node = node.yes if choice else node.no
        chosen = tuple(history)
        prefix = tuple(choice for _, choice in chosen)
        # Prior-tree traversal orders the complete, prefix-free Boolean histories.
        index = bisect_left(
            self.outcomes,
            prefix,
            key=lambda outcome: tuple(choice for _, choice in outcome.history),
        )
        outcome = self.outcomes[index]
        assert outcome.history == chosen
        return outcome


def plan_bernoulli_importance(
    module: QiecModule,
    computation: ComputationId,
    arguments: tuple[int | Fraction, ...],
    direction: tuple[int | Fraction, ...],
    policies: tuple[SiteEstimator, ...],
    *,
    proposal_arguments: tuple[int | Fraction, ...],
    proposal_policies: tuple[SiteEstimator, ...],
    max_paths: int = 4096,
    fuel: int = 50_000,
) -> FiniteImportancePlan:
    """Check an exact finite proposal for a target observation unit.

    The target point supplies the joint observation weights and their
    directional derivatives. The separately supplied proposal point
    supplies the prior branch probabilities, held fixed in that direction.
    Proposal return values do not tilt those probabilities. The check
    compares complete canonical address-outcome histories. In this
    full-support Bernoulli fragment, coverage entails matching the
    support of both finite trees.

    An estimate divides both the target weight and its derivative by the
    same proposal probability. Its exact mean is the raw evidence and
    derivative supplied by ordinary target enumeration.

    Parameters
    ----------
    module : QiecModule
        The actual source used at both points.
    computation : ComputationId
        Its scalar observation computation with one nominal Random row.
    arguments, direction : tuple[int or Fraction, ...]
        The target point and directional perturbation.
    policies : tuple[SiteEstimator, ...]
        An explicit enumerate policy for each reached target site.
    proposal_arguments : tuple[int or Fraction, ...]
        The held-fixed proposal point for the same source parameters.
    proposal_policies : tuple[SiteEstimator, ...]
        An explicit enumerate policy for each reached proposal site.
    max_paths, fuel : int
        Bounds on complete histories at either point and steps per replay.

    Returns
    -------
    FiniteImportancePlan
        Exact coverage and a checked estimator for the unnormalized unit.

    Raises
    ------
    ImportanceCoverageError
        If the proposal misses a positive addressed target history.
    JvpUnsupported
        If a factor is nonpositive or a source or policy leaves the finite
        scalar fragment.
    JvpBoundaryError
        If a parameter-dependent decision is at an equality boundary.
    JvpLimit
        If an arithmetic, path or replay bound is exceeded.
    ValueError
        If a bound, computation identity or input vector is malformed.
    """
    target = _checked_finite_unit(
        module, computation, arguments, direction, policies, max_paths, fuel
    )
    target_atoms = {
        history: _positive_joint(prior, factor, tuple(choice for _, choice in history))
        for history, prior, factor in _prior_histories(target.tree)
    }
    proposal = _checked_finite_unit(
        module,
        computation,
        proposal_arguments,
        (0,) * len(proposal_arguments),
        proposal_policies,
        max_paths,
        fuel,
    )
    proposal_atoms = {
        history: prior.value for history, prior, _ in _prior_histories(proposal.tree)
    }
    for history, atom in target_atoms.items():
        if proposal_atoms.get(history, 0) <= 0:
            raise ImportanceCoverageError(history, atom.value)
    assert target_atoms.keys() == proposal_atoms.keys()

    outcomes: list[ImportanceOutcome] = []
    expected_value = expected_tangent = Fraction(0)
    for history, q in proposal_atoms.items():
        atom = target_atoms[history]
        action = ExactJvp(atom.value / q, atom.tangent / q)
        _bound(action.value, action.tangent)
        expected_value += q * action.value
        expected_tangent += q * action.tangent
        _bound(expected_value, expected_tangent)
        outcomes.append(ImportanceOutcome(history, q, atom, action))
    assert (expected_value, expected_tangent) == (
        target.exact.value,
        target.exact.tangent,
    )
    return FiniteImportancePlan(
        ImportanceCoverage(
            computation,
            tuple(map(_fraction, arguments)),
            tuple(map(_fraction, proposal_arguments)),
            tuple(target_atoms),
        ),
        target.exact,
        tuple(outcomes),
        target.source_steps,
        proposal.source_steps,
        proposal.tree,
    )


__all__ = [
    "FiniteImportancePlan",
    "ImportanceCoverage",
    "ImportanceCoverageError",
    "ImportanceOutcome",
    "plan_bernoulli_importance",
]
