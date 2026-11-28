"""Draw a finite rational law using sequential exact integer choices."""

from fractions import Fraction
from random import Random

from quivers.qiec.differentiation import _bound


def _selector_probabilities(
    probabilities: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    """Compute the sequential conditional probabilities of a categorical law.

    Parameters
    ----------
    probabilities : tuple[Fraction, ...]
        Positive probabilities in declared order, summing to one.

    Returns
    -------
    tuple[Fraction, ...]
        The probabilities of choosing the next position given the earlier
        positions have been skipped. Skipping all selectors gives the last.
    """
    selectors: list[Fraction] = []
    remaining = Fraction(1)
    for probability in probabilities[:-1]:
        selector = probability / remaining
        _bound(selector)
        selectors.append(selector)
        remaining -= probability
    assert remaining == probabilities[-1]
    return tuple(selectors)


def _choose_position(selectors: tuple[Fraction, ...], random: Random) -> int:
    """Select one categorical position by independent integer draws.

    Parameters
    ----------
    selectors : tuple[Fraction, ...]
        Sequential conditional choice probabilities.
    random : Random
        The generator of uniform integer draws.

    Returns
    -------
    int
        The selected original position.
    """
    for position, probability in enumerate(selectors):
        if random.randrange(probability.denominator) < probability.numerator:
            return position
    return len(selectors)
