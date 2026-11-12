# Traced Monoidal

A [trace](https://ncatlab.org/nlab/show/traced+monoidal+category) feeds an output component of a
morphism back into a matching input component [@joyal1996traced]. On finite tensors,
`CartesianTrace` [identifies the two axes and reduces the diagonal](../../guides/categorical.md#traced-monoidal-structure)
with the `Algebra`'s $\oplus$. `IterativeTrace` computes the trace by fixed-point iteration, for
algebras in which it has no closed form. The module also gives the `trace` and `partial_trace`
functions.

::: quivers.categorical.traced
