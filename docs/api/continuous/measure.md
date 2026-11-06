# Measure Algebra

A measure assigns mass to events. These combinators mix, restrict, and transform
measures. Restricting a probability measure to an event of probability 0.2 leaves total
mass 0.2; dividing by 0.2 produces the conditional distribution. Each operation
specifies whether it retains that mass or normalizes it.

See [mass, density, and integration](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample) and
[normalization assumptions on probability kernels](../../semantics/programs.md#5-algebraic-laws-and-their-scope).

For measures and Bayesian updating, see [@bernardo1994bayesian]; for partial Markov categories, see the preprint [@dilavore2025partial].

The compositional measure algebra behind the operator families of the
DSL (`Mixture`, `Restrict`, `Pushforward`, `PointMass`, ...). Each
combinator builds a measure from measures, and
[`normalize_at_boundary`](#quivers.continuous.measure.normalize_at_boundary)
renormalizes a sub-probability measure where it is sampled or scored.

::: quivers.continuous.measure
