"""Single-tape joins of weighted rational relations."""

from __future__ import annotations

from quivers.core.laws import SEMIRING_LAWS, TIMES_COMMUTATIVE
from quivers.core.relations import RationalRelation
from quivers.core.relations.calculus import (
    _JoinLayout,
    _MaterializedWeights,
    _materialize_product,
)
from quivers.core.relations.queries import _validate_limits


def join(
    left: RationalRelation,
    right: RationalRelation,
    *,
    left_tape: int,
    right_tape: int,
    max_states: int,
    max_work: int,
    keep_shared: bool = False,
) -> RationalRelation:
    """Join the operands on one common-type tape.

    Each pair of accepting paths whose words agree on the selected
    tapes contributes once. Surviving input tapes precede output tapes,
    with left tapes before right tapes on either side. With
    `keep_shared`, the common tape occurs once at its left position.
    The weight sources must use the same established commutative
    semiring. At least one tape must remain.

    The limits bound the states, shared-label probes and constructed
    arcs of the sparse product. A join synchronizes a single tape.
    """
    if type(keep_shared) is not bool:
        raise TypeError("keep_shared must be a boolean")
    a, b = left.topology, right.topology
    if (
        type(left_tape) is not int
        or not 0 <= left_tape < len(a.tapes)
        or type(right_tape) is not int
        or not 0 <= right_tape < len(b.tapes)
    ):
        raise ValueError("a relation join requires one valid tape position per operand")
    if a.tapes[left_tape] != b.tapes[right_tape]:
        raise TypeError("a relation join requires the same shared tape type")
    if left.algebra is not right.algebra or not left.algebra.establishes(
        *SEMIRING_LAWS, TIMES_COMMUTATIVE
    ):
        raise TypeError("a relation join requires an established commutative semiring")
    if max_states is None or max_work is None:
        raise ValueError("a relation join requires explicit state and work limits")
    _validate_limits(max_states, max_work)
    layout = _JoinLayout.build(a, b, left_tape, right_tape, keep_shared=keep_shared)
    topology, pairs = _materialize_product(a, b, layout, max_states, max_work)
    return RationalRelation(
        topology, _MaterializedWeights(left, right, pairs), algebra=left.algebra
    )


__all__ = ["join"]
