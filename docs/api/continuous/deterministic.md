# Deterministic Morphisms

A [deterministic kernel](../../guides/programs.md#let-steps) places all its mass at a function’s output. It lets
transformations such as softmax or covariance reconstruction participate in a
[probabilistic pipeline](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram) without introducing another random draw. If the input is random,
the output can still be random across executions; “deterministic” describes the
dependence on the input.

Dirac kernels for deterministic maps: cumulative sums for monotone
splines, the softmax projection onto the simplex, and covariance
reconstruction from a correlation Cholesky factor and scales.

[`LearnedFunction`][quivers.continuous.deterministic.LearnedFunction] is the
learned case: a [parameter source](param_source.md) applied to the input's
coordinates, followed by an output nonlinearity. It is what a
`morphism f : X -> Y [role=function]` declaration compiles to, and a program's
`let z = f(a, b)` calls its
[`apply_values`][quivers.continuous.deterministic.LearnedFunction.apply_values]
with one value per domain factor. Its weights are ordinary module parameters,
fit as point estimates.

::: quivers.continuous.deterministic
