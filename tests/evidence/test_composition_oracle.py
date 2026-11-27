"""Check the epsilon-matching path oracle independently of composition code."""

from collections.abc import Iterator
from fractions import Fraction
from itertools import product
from math import prod

from tests.evidence.oracles import paths


def _interleavings(
    left: tuple[paths.Label, ...],
    right: tuple[paths.Label, ...],
    i: int = 0,
    j: int = 0,
) -> Iterator[paths.Alignment]:
    if i == len(left) and j == len(right):
        yield ()
    if i < len(left) and left[i] is None:
        for tail in _interleavings(left, right, i + 1, j):
            yield ((i, None), *tail)
    if j < len(right) and right[j] is None:
        for tail in _interleavings(left, right, i, j + 1):
            yield ((None, j), *tail)
    if i < len(left) and j < len(right) and left[i] == right[j]:
        for tail in _interleavings(left, right, i + 1, j + 1):
            yield ((i, j), *tail)


# Equation (5) of Allauzen, Riley, and Schalkwyk (2009), using the
# two paths' real arc positions instead of added loops. All live states
# are final. A missing transition blocks the interleaving.
_TRANSITIONS = {
    (0, "symbol"): 0,
    (1, "symbol"): 0,
    (2, "symbol"): 0,
    (0, "paired"): 0,
    (0, "left"): 1,
    (1, "left"): 1,
    (0, "right"): 2,
    (2, "right"): 2,
}


def _accepted(alignment: paths.Alignment, left: tuple[paths.Label, ...]) -> bool:
    state = 0
    for i, j in alignment:
        if i is None:
            event = "right"
        elif j is None:
            event = "left"
        else:
            event = "paired" if left[i] is None else "symbol"
        state = _TRANSITIONS.get((state, event), -1)
        if state == -1:
            return False
    return True


def test_three_state_filter_accepts_one_alignment_per_compatible_label_pair():
    labels = (None, "a", "b")
    sequences = tuple(
        sequence for length in range(5) for sequence in product(labels, repeat=length)
    )
    for left, right in product(sequences, repeat=2):
        candidates = [
            alignment
            for alignment in _interleavings(left, right)
            if _accepted(alignment, left)
        ]
        expected = paths.epsilon_matching_alignment(left, right)
        assert candidates == ([] if expected is None else [expected]), (left, right)

    # Counting gives a strict multiplicity witness. Epsilon shifts leave
    # both outer strings and the nonzero product of weights unchanged.
    unfiltered = tuple(_interleavings((None,), (None,)))
    assert len(unfiltered) == 3
    assert [step for step in unfiltered if _accepted(step, (None,))] == [((0, 0),)]


def test_compatible_pairs_preserve_parallel_paths_and_shared_weight_occurrences():
    left = paths.Machine(
        arcs=(
            paths.Arc(
                name="l1", source=0, target=1, labels=("x", None), weight="shared"
            ),
            paths.Arc(
                name="l2", source=0, target=1, labels=("x", None), weight="shared"
            ),
            paths.Arc(name="lm", source=1, target=2, labels=(None, "a"), weight="left"),
            paths.Arc(name="lt", source=2, target=3, labels=("q", None), weight="zero"),
            paths.Arc(name="lb", source=0, target=4, labels=("x", "b"), weight="unit"),
        ),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=tuple(paths.Mark(state=state, weight="unit") for state in (0, 2, 3, 4)),
    )
    right = paths.Machine(
        arcs=(
            paths.Arc(
                name="r1", source=0, target=1, labels=(None, "z"), weight="shared"
            ),
            paths.Arc(
                name="rm", source=1, target=2, labels=("a", None), weight="right"
            ),
            paths.Arc(name="rt", source=2, target=3, labels=(None, "r"), weight="tail"),
        ),
        initial=(paths.Mark(state=0, weight="unit"),),
        final=tuple(paths.Mark(state=state, weight="unit") for state in (0, 2, 3)),
    )
    weight = {
        "shared": Fraction(2, 3),
        "unit": Fraction(1),
        "left": Fraction(2),
        "right": Fraction(3),
        "zero": Fraction(0),
        "tail": Fraction(1, 2),
    }
    pairs = list(paths.compatible_path_pairs(left, right))
    assert len(pairs) == 9
    values: dict[paths.Strings, Fraction] = {}
    derivatives: dict[tuple[paths.Strings, str], Fraction] = {}
    for left_path, right_path, alignment in pairs:
        word_in, _ = left_path.strings() or ((), ())
        _, word_out = right_path.strings() or ((), ())
        key = word_in, word_out
        assert [i for i, _ in alignment if i is not None] == list(
            range(len(left_path.arcs))
        )
        assert [j for _, j in alignment if j is not None] == list(
            range(len(right_path.arcs))
        )
        factors = left_path.factors() + right_path.factors()
        values[key] = values.get(key, Fraction(0)) + prod(
            weight[name] for name in factors
        )
        for i, name in enumerate(factors):
            slot = key, name
            derivatives[slot] = derivatives.get(slot, Fraction(0)) + prod(
                weight[other] for j, other in enumerate(factors) if i != j
            )
    assert values == paths.compose_relations(
        paths.relation(left, weight, tapes=2), paths.relation(right, weight, tapes=2)
    )
    assert values[((), ())] == 1
    assert values[(("x", "q"), ("z",))] == 0
    assert derivatives[((("x",), ("z",)), "shared")] == 16
    assert derivatives[((("x", "q"), ("z",)), "zero")] == Fraction(16, 3)
