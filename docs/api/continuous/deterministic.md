# Deterministic Morphisms

A [deterministic kernel](../../guides/programs.md#let-steps) places all its mass at a function’s output. It lets
transformations such as softmax or covariance reconstruction participate in a
[probabilistic pipeline](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram) without introducing another random draw. If the input is random,
the output can still be random across executions; “deterministic” describes the
dependence on the input.

Dirac kernels for deterministic maps: cumulative sums for monotone
splines, the softmax projection onto the simplex, and covariance
reconstruction from a correlation Cholesky factor and scales.

::: quivers.continuous.deterministic
