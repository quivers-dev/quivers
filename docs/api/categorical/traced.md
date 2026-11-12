# Traced Monoidal

A [trace](https://ncatlab.org/nlab/show/traced+monoidal+category) feeds an output component of a
morphism back into a matching input component [@joyal1996traced]. On finite tensors,
`CompactClosedTrace` [identifies the two axes and reduces the diagonal](../../guides/categorical.md#traced-monoidal-structure)
with the `Algebra`'s $\oplus$. It is the trace that compact closure induces, and it satisfies the
trace axioms when the `Algebra` is a `Semiring`, which its constructor requires. The module also
gives the `trace` and `partial_trace` functions.

::: quivers.categorical.traced
