"""Exact ordered multinomial ancestry for a checked, finite-step objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from random import Random

from quivers.qiec._finite_choices import _choose_position, _selector_probabilities
from quivers.qiec.differentiation import (
    ExactJvp,
    JvpLimit,
    JvpUnsupported,
    _bound,
    _dual,
    _Dual,
    _fraction,
    _prepare_scalar,
    _run_derivative,
    _ScalarDerivativeEvaluator,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.laws import _validate_limit
from quivers.qiec.module import QiecModule


@dataclass(frozen=True, slots=True)
class ResamplingParticle:
    """The exact represented value and weight at one particle position.

    Parameters
    ----------
    value : ExactJvp
        Its scalar value and directional derivative.
    weight : ExactJvp
        Its positive unnormalized weight and directional derivative.
    """

    value: ExactJvp
    weight: ExactJvp


@dataclass(frozen=True, slots=True)
class ResamplingAncestor:
    """One position with its categorical probability and score.

    Parameters
    ----------
    position : int
        The original zero-based position in the ordered particle table.
    value : ExactJvp
        The scalar and directional derivative at this position.
    probability : ExactJvp
        The normalized probability of choosing this position and its derivative.
    score : Fraction
        The exact directional derivative of its log probability.
    """

    position: int
    value: ExactJvp
    probability: ExactJvp
    score: Fraction


@dataclass(frozen=True, slots=True)
class ResamplingOutcome:
    """The native finite objective and its estimate for one ordered ancestry.

    Parameters
    ----------
    positions : tuple[int, ...]
        The original ancestor position selected for each offspring slot.
    probability : ExactJvp
        The mass and derivative of this whole ordered ancestry.
    objective : ExactJvp
        The checked objective and its directional derivative for these
        fixed positions.
    ancestry_score : Fraction
        The sum of the selected log probability derivatives across slots.
    estimate : ExactJvp
        The resulting sampled objective and derivative estimate including
        its ancestry contribution.
    """

    positions: tuple[int, ...]
    probability: ExactJvp
    objective: ExactJvp
    ancestry_score: Fraction
    estimate: ExactJvp


@dataclass(frozen=True, slots=True)
class MultinomialResamplingPlan:
    """A complete exact finite multinomial stage for the declared objective.

    Parameters
    ----------
    objective : ComputationId
        The actual checked scalar source computation.
    draws : int
        The number of independent ordered offspring slots.
    total_weight : ExactJvp
        The original particle weight sum and its derivative.
    exact : ExactJvp
        The expected finite-step objective and its directional derivative.
    ancestors : tuple[ResamplingAncestor, ...]
        The normalized table of original particle positions.
    outcomes : tuple[ResamplingOutcome, ...]
        Every ordered assignment with its exact probability and estimate.
    source_steps : int
        The total measured reference work over the checked objective runs.
    selector_probabilities : tuple[Fraction, ...]
        The exact sequential conditional probabilities for one categorical
        draw. The last position is the complement of all earlier choices.
    """

    objective: ComputationId
    draws: int
    total_weight: ExactJvp
    exact: ExactJvp
    ancestors: tuple[ResamplingAncestor, ...]
    outcomes: tuple[ResamplingOutcome, ...]
    source_steps: int
    selector_probabilities: tuple[Fraction, ...] = field(repr=False)

    def sample(self, *, seed: int | None = None) -> ResamplingOutcome:
        """Choose one complete ancestry from the exact multinomial law.

        Parameters
        ----------
        seed : int or None
            An optional seed for independent exact integer choices.

        Returns
        -------
        ResamplingOutcome
            The sampled ordered positions and their raw objective estimate.
        """
        random = Random(seed)
        count = len(self.ancestors)
        index = 0
        for _ in range(self.draws):
            position = _choose_position(self.selector_probabilities, random)
            index = index * count + position
        return self.outcomes[index]


def _ancestor_table(
    particles: tuple[ResamplingParticle, ...],
) -> tuple[tuple[ResamplingAncestor, ...], ExactJvp, tuple[Fraction, ...]]:
    """Validate and normalize a finite table without coalescing its positions.

    Parameters
    ----------
    particles : tuple[ResamplingParticle, ...]
        The ordered nonempty particle table with positive exact weights.

    Returns
    -------
    tuple
        Ancestors, the total input weight and categorical selectors.

    Raises
    ------
    JvpUnsupported
        If a weight on the declared support is nonpositive.
    ValueError
        If a represented scalar is outside the exact carrier.
    """
    entries = tuple(
        ResamplingParticle(
            ExactJvp(_fraction(item.value.value), _fraction(item.value.tangent)),
            ExactJvp(_fraction(item.weight.value), _fraction(item.weight.tangent)),
        )
        for item in particles
    )
    value = tangent = Fraction(0)
    for entry in entries:
        weight = entry.weight
        if weight.value <= 0:
            raise JvpUnsupported(
                "finite multinomial resampling needs positive particle weights"
            )
        _bound(entry.value.value, entry.value.tangent, weight.value, weight.tangent)
        value += weight.value
        tangent += weight.tangent
        _bound(value, tangent)
    total = ExactJvp(value, tangent)
    ancestors: list[ResamplingAncestor] = []
    for index, entry in enumerate(entries):
        weight = entry.weight
        p = weight.value / value
        dp = (weight.tangent - p * tangent) / value
        score = weight.tangent / weight.value - tangent / value
        _bound(p, dp, score)
        assert p * score == dp
        ancestors.append(ResamplingAncestor(index, entry.value, ExactJvp(p, dp), score))
    selectors = _selector_probabilities(
        tuple(ancestor.probability.value for ancestor in ancestors)
    )
    return tuple(ancestors), total, selectors


def plan_multinomial_resampling(
    module: QiecModule,
    objective: str,
    particles: tuple[ResamplingParticle, ...],
    *,
    arguments: tuple[int | Fraction, ...] = (),
    direction: tuple[int | Fraction, ...] = (),
    max_outcomes: int = 4096,
    fuel: int = 65_536,
) -> MultinomialResamplingPlan:
    """Check one finite, ordered multinomial resampling stage.

    The closed scalar objective takes one selected value per offspring
    slot, followed by any direct arguments. The particle jets and direct
    argument directions supply its fixed-ancestry directional action.
    Strict positivity preserves the normalized support locally. Every
    possible ordered ancestry must pass the exact scalar source checks.

    The derivative estimate adds the objective times the sum of the
    ancestor scores to that directional action. The sum counts each
    offspring slot, including repeated choices of the same position.
    The exact result integrates the declared finite-step objective.

    Parameters
    ----------
    module : QiecModule
        The source of the declared scalar objective.
    objective : str
        The name of the closed scalar objective with at least one slot.
    particles : tuple[ResamplingParticle, ...]
        The ordered, nonempty input table of value and weight jets.
    arguments, direction : tuple[int or Fraction, ...]
        The exact direct arguments and directions after the offspring
        slots in the objective's positional parameters.
    max_outcomes : int
        The maximum number of complete ordered assignments.
    fuel : int
        The reference step bound for each objective evaluation.

    Returns
    -------
    MultinomialResamplingPlan
        The checked ancestry law, exact objective and sampled estimator.

    Raises
    ------
    ValueError
        If inputs, objective arity or a bound is malformed.
    JvpUnsupported
        If weights are nonpositive or a reached source action is unsupported.
    JvpBoundaryError
        If an admitted objective branch meets an input-dependent equality.
    JvpLimit
        If assignment count, arithmetic or reference steps exceed a bound.
    """
    _validate_limit("max_outcomes", max_outcomes)
    if not particles:
        raise ValueError(
            "finite multinomial resampling needs a nonempty particle table"
        )
    if len(arguments) != len(direction):
        raise ValueError("direct arguments and direction must have equal arity")
    registry, declaration = _prepare_scalar(module, objective, fuel)
    draws = len(declaration.parameters) - len(arguments)
    if draws < 1:
        raise ValueError("the resampling objective needs at least one offspring slot")
    count = 1
    for _ in range(draws):
        count *= len(particles)
        if count > max_outcomes:
            raise JvpLimit(
                f"multinomial resampling exceeds {max_outcomes} ordered assignments"
            )
    ancestors, total_weight, selectors = _ancestor_table(particles)
    direct = tuple(
        _dual(_fraction(value), _fraction(tangent), True)
        for value, tangent in zip(arguments, direction, strict=True)
    )
    seeds = tuple(
        _dual(ancestor.value.value, ancestor.value.tangent, True)
        for ancestor in ancestors
    )
    outcomes: list[ResamplingOutcome] = []
    mean = derivative = expected_estimator = Fraction(0)
    mass_total = mass_tangent = Fraction(0)
    source_steps = 0
    for positions in product(range(len(ancestors)), repeat=draws):
        evaluator = _ScalarDerivativeEvaluator(module=module, fuel=fuel)
        positional = tuple(seeds[position] for position in positions)
        environment = dict(
            zip(declaration.parameters, (*positional, *direct), strict=True)
        )
        source = _run_derivative(evaluator, declaration, registry, environment)
        assert isinstance(source, _Dual)
        source_steps += evaluator._steps
        mass, slope, score = Fraction(1), Fraction(0), Fraction(0)
        for position in positions:
            ancestor = ancestors[position]
            p = ancestor.probability
            slope = slope * p.value + mass * p.tangent
            mass *= p.value
            score += ancestor.score
            _bound(mass, slope, score)
        assert slope == mass * score
        cost = ExactJvp(source.value, source.tangent)
        estimate = ExactJvp(cost.value, cost.tangent + cost.value * score)
        _bound(estimate.value, estimate.tangent)
        mean += mass * cost.value
        derivative += slope * cost.value + mass * cost.tangent
        expected_estimator += mass * estimate.tangent
        mass_total += mass
        mass_tangent += slope
        _bound(mean, derivative, expected_estimator, mass_total, mass_tangent)
        outcomes.append(
            ResamplingOutcome(positions, ExactJvp(mass, slope), cost, score, estimate)
        )
    assert derivative == expected_estimator and mass_total == 1 and mass_tangent == 0
    return MultinomialResamplingPlan(
        declaration.id,
        draws,
        total_weight,
        ExactJvp(mean, derivative),
        ancestors,
        tuple(outcomes),
        source_steps,
        selectors,
    )


__all__ = [
    "MultinomialResamplingPlan",
    "ResamplingAncestor",
    "ResamplingOutcome",
    "ResamplingParticle",
    "plan_multinomial_resampling",
]
