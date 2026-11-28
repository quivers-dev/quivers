"""Exact first-order inclusion weights for fixed finite unit subsampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from random import Random

from torch import Tensor

from quivers.qiec._finite_choices import _choose_position, _selector_probabilities
from quivers.qiec.differentiation import (
    ExactJvp,
    JvpLimit,
    JvpUnsupported,
    _bound,
    _fraction,
)
from quivers.qiec.laws import _validate_limit


@dataclass(frozen=True, slots=True)
class UnitSubset:
    """One sample of whole unit positions with its ordinary design probability.

    Parameters
    ----------
    positions : tuple[int, ...]
        The strictly increasing zero-based positions of the units kept.
    probability : Fraction
        The fixed probability of drawing this complete subset.
    """

    positions: tuple[int, ...]
    probability: Fraction

    @classmethod
    def from_plate_rows(
        cls, plate: str, rows: Tensor, probability: int | Fraction
    ) -> UnitSubset:
        """Read the existing plate row format for an axis of whole units.

        Parameters
        ----------
        plate : str
            The axis name used by the plate subsampling handler.
        rows : Tensor
            Its ordinary subsample index. Positions must identify whole units.
        probability : int or Fraction
            The fixed probability of this complete subset.

        Returns
        -------
        UnitSubset
            The validated positions and exact design probability.

        Raises
        ------
        ValueError
            If the row format or probability is malformed.
        """
        from quivers.effects.subsample import SubsampleHandler

        selection = SubsampleHandler(plate, rows).index
        return cls(tuple(selection.tolist()), _fraction(probability))


@dataclass(frozen=True, slots=True)
class UnitSubsampleEstimate:
    """The raw total estimate and derivative on one finite whole-unit subset.

    Parameters
    ----------
    subset : UnitSubset
        The actual positions and probability used by the design.
    estimate : ExactJvp
        The inverse-inclusion weighted sum of selected unit jets.
    """

    subset: UnitSubset
    estimate: ExactJvp


@dataclass(frozen=True, slots=True)
class FiniteUnitSubsamplingPlan:
    """A finite design verified for a raw additive unit objective.

    Parameters
    ----------
    exact : ExactJvp
        The full population total and its directional derivative.
    inclusion_probabilities : tuple[Fraction, ...]
        Probability of including each original unit position.
    outcomes : tuple[UnitSubsampleEstimate, ...]
        Every declared subset and its exact weighted action.
    work : int
        Total unit positions and subset memberships checked.
    selector_probabilities : tuple[Fraction, ...]
        The exact sequential conditional probabilities used to draw a subset.
    """

    exact: ExactJvp
    inclusion_probabilities: tuple[Fraction, ...]
    outcomes: tuple[UnitSubsampleEstimate, ...]
    work: int
    selector_probabilities: tuple[Fraction, ...] = field(repr=False)

    def sample(self, *, seed: int | None = None) -> UnitSubsampleEstimate:
        """Draw one subset from the declared exact design.

        Parameters
        ----------
        seed : int or None
            An optional seed for independent uniform integer draws.

        Returns
        -------
        UnitSubsampleEstimate
            The selected whole units and their raw weighted estimate.
        """
        selected = _choose_position(self.selector_probabilities, Random(seed))
        return self.outcomes[selected]


def plan_unit_subsampling(
    units: tuple[ExactJvp, ...],
    design: tuple[UnitSubset, ...],
    *,
    max_outcomes: int = 4096,
    max_work: int = 250_000,
) -> FiniteUnitSubsamplingPlan:
    """Check exact subsampling of a fixed finite sum of whole unit returns.

    Each input is an exact scalar contribution and its directional
    derivative, such as an exact source JVP or a finite marginalized
    unit's raw normalizer. The design is fixed in that direction. The
    probability of including a unit is the sum of probabilities of the
    declared subsets containing its position. Every unit needs positive
    inclusion probability.

    Dividing each selected value and derivative by that probability
    makes the two expected sums equal their full-population totals.
    This is an additive objective. A log likelihood score for a unit can
    enter as a separately defined additive contribution.

    Parameters
    ----------
    units : tuple[ExactJvp, ...]
        A nonempty ordered table containing one exact jet per whole unit.
    design : tuple[UnitSubset, ...]
        The full, fixed probability law of nonempty sorted subsets.
    max_outcomes : int
        The maximum number of distinct declared subsets.
    max_work : int
        The maximum total of unit positions and subset memberships.

    Returns
    -------
    FiniteUnitSubsamplingPlan
        The checked inclusion probabilities and raw total estimator.

    Raises
    ------
    ValueError
        If the whole unit table, subset law or a limit is malformed.
    JvpUnsupported
        If a unit is absent from the positive-probability design.
    JvpLimit
        If the checked subset count, memberships or arithmetic exceed a bound.
    """
    _validate_limit("max_outcomes", max_outcomes)
    _validate_limit("max_work", max_work)
    if not units or not design:
        raise ValueError("unit subsampling needs nonempty units and subset design")
    if len(design) > max_outcomes:
        raise JvpLimit(f"unit subsampling exceeds {max_outcomes} distinct subsets")
    work = len(units) + sum(len(item.positions) for item in design)
    if work > max_work:
        raise JvpLimit(
            f"unit subsampling exceeds {max_work} unit positions and memberships"
        )

    atoms: list[UnitSubset] = []
    inclusion = [Fraction(0)] * len(units)
    seen: set[tuple[int, ...]] = set()
    probability_total = Fraction(0)
    for subset in design:
        positions = subset.positions
        if (
            not isinstance(positions, tuple)
            or not positions
            or any(
                type(index) is not int or not 0 <= index < len(units)
                for index in positions
            )
            or any(left >= right for left, right in zip(positions, positions[1:]))
        ):
            raise ValueError(
                "a unit subset needs distinct positions in increasing order within the population"
            )
        if positions in seen:
            raise ValueError("the unit design declares the same subset twice")
        seen.add(positions)
        probability = _fraction(subset.probability)
        if not 0 < probability <= 1:
            raise ValueError(
                "unit subset probabilities must be strictly positive and at most one"
            )
        _bound(probability)
        probability_total += probability
        _bound(probability_total)
        for index in positions:
            inclusion[index] += probability
            _bound(inclusion[index])
        atoms.append(UnitSubset(positions, probability))
    if probability_total != 1:
        raise ValueError("unit subset probabilities must sum to one")
    missing = next(
        (index for index, probability in enumerate(inclusion) if not probability), None
    )
    if missing is not None:
        raise JvpUnsupported(
            f"whole unit position {missing} has zero inclusion probability"
        )

    expected = expected_tangent = total = tangent_total = Fraction(0)
    actions: list[ExactJvp] = []
    for unit, probability in zip(units, inclusion, strict=True):
        value, tangent = _fraction(unit.value), _fraction(unit.tangent)
        _bound(value, tangent)
        total += value
        tangent_total += tangent
        action = ExactJvp(value / probability, tangent / probability)
        _bound(total, tangent_total, action.value, action.tangent)
        actions.append(action)
    outputs: list[UnitSubsampleEstimate] = []
    for subset in atoms:
        value = tangent = Fraction(0)
        for index in subset.positions:
            value += actions[index].value
            tangent += actions[index].tangent
            _bound(value, tangent)
        expected += subset.probability * value
        expected_tangent += subset.probability * tangent
        _bound(expected, expected_tangent)
        outputs.append(UnitSubsampleEstimate(subset, ExactJvp(value, tangent)))
    assert (expected, expected_tangent) == (total, tangent_total)
    return FiniteUnitSubsamplingPlan(
        ExactJvp(total, tangent_total),
        tuple(inclusion),
        tuple(outputs),
        work,
        _selector_probabilities(tuple(subset.probability for subset in atoms)),
    )


__all__ = [
    "FiniteUnitSubsamplingPlan",
    "UnitSubsampleEstimate",
    "UnitSubset",
    "plan_unit_subsampling",
]
