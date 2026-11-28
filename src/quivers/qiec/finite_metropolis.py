"""Exact directional derivatives of finite-step Metropolis--Hastings laws."""

from __future__ import annotations

from fractions import Fraction

from quivers.qiec.differentiation import ExactJvp, JvpBoundaryError, JvpLimit


def _masses(entries: tuple[ExactJvp, ...], name: str, *, normalize: bool) -> None:
    """Validate pointwise conditions on a differentiable finite measure.

    Parameters
    ----------
    entries : tuple[ExactJvp, ...]
        The exact mass and directional action of every coordinate.
    name : str
        The input label used in diagnostics.
    normalize : bool
        Whether the input must have unit mass and zero total tangent.

    Raises
    ------
    ValueError
        If an entry is outside the admitted nonnegative measure domain,
        or the required normalization and its derivative fail.
    """
    for entry in entries:
        if not isinstance(entry, ExactJvp) or any(
            type(coordinate) not in (int, Fraction)
            for coordinate in (entry.value, entry.tangent)
        ):
            raise ValueError(f"{name} requires exact rational value/tangent pairs")
        if entry.value < 0 or (entry.value == 0 and entry.tangent != 0):
            raise ValueError(f"{name} leaves nonnegative mass at the supplied point")
    if normalize and (
        sum(entry.value for entry in entries) != 1
        or sum(entry.tangent for entry in entries) != 0
    ):
        raise ValueError(f"{name} requires unit mass and zero total tangent")


def _move(
    current: ExactJvp, proposed: ExactJvp, forward: ExactJvp, reverse: ExactJvp
) -> ExactJvp:
    """Differentiate the smaller of the two admissible off-diagonal masses.

    Parameters
    ----------
    current, proposed : ExactJvp
        Strictly positive target masses at the endpoints.
    forward, reverse : ExactJvp
        Forward and reverse proposal probabilities.

    Returns
    -------
    ExactJvp
        The off-diagonal transition mass and its directional derivative.

    Raises
    ------
    JvpBoundaryError
        If the two minimum branches have different slopes at equality.
    """
    ratio_mass = Fraction(proposed.value * reverse.value, current.value)
    if forward.value < ratio_mass:
        return ExactJvp(Fraction(forward.value), Fraction(forward.tangent))
    ratio_tangent = Fraction(
        proposed.tangent * reverse.value
        + proposed.value * reverse.tangent
        - ratio_mass * current.tangent,
        current.value,
    )
    if forward.value == ratio_mass and forward.tangent != ratio_tangent:
        raise JvpBoundaryError(
            "Metropolis acceptance tie has unequal directional slopes"
        )
    return ExactJvp(ratio_mass, ratio_tangent)


def finite_metropolis_jvp(
    initial: tuple[ExactJvp, ...],
    target: tuple[ExactJvp, ...],
    proposal: tuple[tuple[ExactJvp, ...], ...],
    *,
    steps: int,
    max_work: int = 65_536,
) -> tuple[ExactJvp, ...]:
    """Return the exact law and directional derivative after a fixed number of steps.

    Inputs supply values and directional derivatives of differentiable
    curves through the point. The initial and proposal curves are
    normalized and nonnegative in a neighborhood. The target curve is
    positive and may be unnormalized. Equal acceptance values have a
    directional derivative when their directional slopes agree. The
    recurrence propagates both rejection and acceptance derivatives.

    Parameters
    ----------
    initial : tuple[ExactJvp, ...]
        The initial law and its tangent in finite-state order.
    target : tuple[ExactJvp, ...]
        The positive target masses and their tangents in the same order.
    proposal : tuple[tuple[ExactJvp, ...], ...]
        For each source state, its normalized proposal row and tangent.
    steps : int
        The exact number of transitions after initialization.
    max_work : int
        The maximum number of table positions for validation, transition
        construction and recurrence steps, measured as an arithmetic
        work bound independent of coefficient bit length.

    Returns
    -------
    tuple[ExactJvp, ...]
        One mass and tangent per state in the finite-step law.

    Raises
    ------
    ValueError
        If the dimensions, limits, or pointwise measure conditions fail.
    JvpBoundaryError
        If an acceptance tie has unequal slopes along the supplied direction.
    JvpLimit
        If the arithmetic work bound would be exceeded.
    """
    if type(steps) is not int or steps < 0:
        raise ValueError("Metropolis steps must be a nonnegative integer")
    if type(max_work) is not int or max_work < 1:
        raise ValueError("Metropolis max_work must be a positive integer")
    size = len(initial)
    if (
        size == 0
        or len(target) != size
        or len(proposal) != size
        or any(len(row) != size for row in proposal)
    ):
        raise ValueError("Metropolis inputs must share one nonempty finite state space")
    work = 2 * size + (1 + int(steps > 0) + steps) * size * size
    if work > max_work:
        raise JvpLimit(
            f"Metropolis law needs {work} work positions, exceeding {max_work}"
        )
    _masses(initial, "initial law", normalize=True)
    _masses(target, "target", normalize=False)
    if any(entry.value == 0 for entry in target):
        raise ValueError("Metropolis target masses must all be strictly positive")
    for i, row in enumerate(proposal):
        _masses(row, f"proposal row {i}", normalize=True)

    state = tuple(ExactJvp(Fraction(m.value), Fraction(m.tangent)) for m in initial)
    if steps == 0:
        return state
    transition: list[list[ExactJvp]] = []
    zero = ExactJvp(Fraction(0), Fraction(0))
    for i, current in enumerate(target):
        row = [
            zero if i == j else _move(current, proposed, proposal[i][j], proposal[j][i])
            for j, proposed in enumerate(target)
        ]
        row[i] = ExactJvp(
            Fraction(1) - sum(entry.value for entry in row),
            -sum((entry.tangent for entry in row), Fraction(0)),
        )
        transition.append(row)
    for _ in range(steps):
        following: list[ExactJvp] = []
        for column in zip(*transition, strict=True):
            value, tangent = Fraction(0), Fraction(0)
            for mass, move in zip(state, column, strict=True):
                value += mass.value * move.value
                tangent += mass.tangent * move.value + mass.value * move.tangent
            following.append(ExactJvp(value, tangent))
        state = tuple(following)
    return state


__all__ = ["finite_metropolis_jvp"]
