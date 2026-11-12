"""The algebra requirement of the enriched constructions.

Ends, coends, profunctor composition, weighted (co)limits, the Yoneda
lemma and Day convolution are joins, meets and internal homs in a
complete lattice; they are those universal constructions only when the
algebra's join is the lattice join, its meet the infimum and its
residuum the internal hom, which the `IdempotentSemiring` instances
guarantee and the test suite checks.
"""

from __future__ import annotations

from quivers.core.algebras import CompositionRule, IdempotentSemiring


def require_lattice(algebra: CompositionRule | None, what: str) -> IdempotentSemiring:
    """Return ``algebra`` as an `IdempotentSemiring`, or raise.

    Parameters
    ----------
    algebra : CompositionRule or None
        The algebra a construction was asked to use.
    what : str
        The construction, named in the error.

    Returns
    -------
    IdempotentSemiring
        ``algebra``.

    Raises
    ------
    TypeError
        If ``algebra`` is not an `IdempotentSemiring`.
    """
    if not isinstance(algebra, IdempotentSemiring):
        raise TypeError(
            f"{what} is a construction in a complete lattice and needs an "
            f"IdempotentSemiring algebra (BOOLEAN, GODEL, TROPICAL, MAX_PLUS, "
            f"or a dual of BOOLEAN or GODEL); got {algebra!r}"
        )
    return algebra
