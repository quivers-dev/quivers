"""First-class transformation composition.

A *transformation* in quivers is either a
`quivers.core.algebra_morphisms.CarrierMap`
(pointwise, lax monoidal) or a
`quivers.core.morphism_transformations.MorphismTransformation`
(shape-aware).  Both expose a ``source`` and ``target`` algebra
and an ``apply`` that consumes a morphism's tensor.

This module gives them a *composition* operation: ``compose_trans(t1, t2, ...)``
returns a `TransSeq` value that applies the steps in order
when handed to `Morphism.change_base`.  Adjacent ``target``
and ``source`` are type-checked at compose time.  Nested
`TransSeq` are flattened so the resulting steps tuple is
always a single flat sequence of base transformations.

This is the Python-side surface for the DSL's ``t1 >>> t2``
operator: in the ``.qvr`` surface,
``let pipe = softmax(B) >>> expectation`` builds the same value
that ``compose_trans(softmax(B), MARKOV_TO_PRODUCT_FUZZY)`` builds here.
"""

from __future__ import annotations

from quivers.core.morphism_transformations import MorphismTransformation
from quivers.core.algebra_morphisms import CarrierMap


type TransValue = CarrierMap | MorphismTransformation
"""A single base transformation: either a
`CarrierMap` (pointwise) or a
`MorphismTransformation` (shape-aware)."""


class TransSeq:
    """Sequence of transformations applied left-to-right.

    Constructed by `compose_trans`; consumed by
    `Morphism.change_base`.  Each adjacent ``target`` /
    ``source`` boundary in `steps` is verified at
    construction time, so `apply` does not have to
    revalidate.

    A ``TransSeq`` itself exposes the same ``source`` / ``target``
    / ``name`` interface as a single transformation, so it can be
    fed into further `compose_trans` calls and the nesting
    is flattened.
    """

    __slots__ = ("_steps", "_source", "_target")

    def __init__(self, steps: tuple[TransValue, ...]) -> None:
        if len(steps) < 2:
            raise ValueError(
                "TransSeq: requires at least two steps; a "
                "single-step trans value is the base "
                "transformation itself, not a sequence"
            )
        for i in range(len(steps) - 1):
            tgt = steps[i].target
            src = steps[i + 1].source
            if type(tgt) is not type(src):
                raise TypeError(
                    f"compose_trans: target of step {i} "
                    f"({tgt.name!r}) does not match source of "
                    f"step {i + 1} ({src.name!r})"
                )
        self._steps: tuple[TransValue, ...] = steps
        self._source = steps[0].source
        self._target = steps[-1].target

    @property
    def steps(self) -> tuple[TransValue, ...]:
        return self._steps

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def name(self) -> str:
        return " >>> ".join(s.name for s in self._steps)

    def __repr__(self) -> str:
        return f"TransSeq({self.name})"


def compose_trans(*steps: TransValue | TransSeq) -> TransSeq:
    """Compose two or more transformations into a single sequence.

    Each argument is either a base transformation (a
    ``CarrierMap`` or ``MorphismTransformation``) or
    another `TransSeq` whose steps are inlined.  The
    resulting sequence is flat; nested compositions never appear
    in `TransSeq.steps`.

    Parameters
    ----------
    *steps : TransValue or TransSeq
        Two or more transformations.

    Returns
    -------
    TransSeq
        The composed sequence with ``source = steps[0].source``
        and ``target = steps[-1].target``.

    Raises
    ------
    ValueError
        If fewer than two steps are supplied.
    TypeError
        If two adjacent steps' target / source algebras differ.
    """
    if len(steps) < 2:
        raise ValueError("compose_trans: requires at least two transformations")
    flat: list[TransValue] = []
    for step in steps:
        if isinstance(step, TransSeq):
            flat.extend(step.steps)
        else:
            flat.append(step)
    return TransSeq(tuple(flat))


__all__ = ["TransSeq", "TransValue", "compose_trans"]
