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
[@malherbe2012partially]. It accepts the Boolean, Counting, Markov,
log-probability, and max-plus semirings, with float32 or float64 matrices in
their declared carriers. An undefined closure raises
[`DivergentClosureError`](#quivers.categorical.traced.DivergentClosureError).

::: quivers.categorical.traced
