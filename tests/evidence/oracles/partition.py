"""The eight proof occurrences for S over the sentence a b a."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from math import prod

PROOFS = (
    ("((ab)a)", "a_s", "a_s"),
    ("((ab)a)", "a_s", "a_v"),
    ("((ab)a)", "a_v", "a_s"),
    ("((ab)a)", "a_v", "a_v"),
    ("(a(ba))", "a_s", "a_s"),
    ("(a(ba))", "a_s", "a_v"),
    ("(a(ba))", "a_v", "a_s"),
    ("(a(ba))", "a_v", "a_v"),
)
"""The bracketing and the lexical row at each of the two a positions."""


def partition(
    weights: Mapping[str, Fraction],
) -> tuple[Fraction, dict[str, Fraction] | None]:
    """Enumerate linear proof masses and their normalized use counts.

    Each tree uses join twice and consumes the middle b through lift.
    An a can be S directly or use lift after the V lexical row. For a
    positive partition, the derivative of log mass in a log weight is
    the sum of the proof's mass times its use count, divided by the mass.
    """
    mass = Fraction(0)
    uses = dict.fromkeys(weights, Fraction(0))
    for _, left, right in PROOFS:
        factors = ("join", "join", "b_v", "lift", left, right) + (
            ("lift",) * (int(left == "a_v") + int(right == "a_v"))
        )
        term = prod((weights[name] for name in factors), start=Fraction(1))
        mass += term
        for name in factors:
            uses[name] += term
    return mass, {name: use / mass for name, use in uses.items()} if mass else None
