# Traced Monoidal

A [trace](https://ncatlab.org/nlab/show/traced+monoidal+category) feeds an output component of a
morphism back into a matching input component [@joyal1996traced]. On finite tensors,
`CompactClosedTrace` [identifies the two axes and reduces the diagonal](../../guides/categorical.md#traced-monoidal-structure)
with the `Algebra`'s $\oplus$. It is the trace that compact closure induces, and it satisfies the
trace axioms when the `Algebra` is a `Semiring`, which its constructor requires. The module also
gives the `trace` and `partial_trace` functions.

[`KleeneStarTrace`](#quivers.categorical.traced.KleeneStarTrace) takes coproducts
as its monoidal product and sums paths through a feedback block. It is a
[partial trace](../../guides/categorical.md#coproduct-feedback): its method is
defined only when the entire feedback closure is finite
[@malherbe2012partially]. It accepts a semiring by its capability: any
[`Semiring`][quivers.core.algebras.Semiring] with a [`Star`][quivers.core.algebras.Star] that establishes the
least-solution law, which includes the Boolean, Gödel, tropical, max-plus, log,
counting, and Markov semirings. Matrices are float32 or float64 in the
algebra's carrier. An undefined closure raises
[`DivergentClosureError`](#quivers.categorical.traced.DivergentClosureError),
and a loop that rounding cannot decide raises
[`UndecidedClosureError`][quivers.core.algebras.UndecidedClosureError].

::: quivers.categorical.traced
