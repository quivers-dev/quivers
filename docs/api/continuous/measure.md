# Measure Algebra

A measure assigns [mass to events](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample). These combinators mix, restrict, and transform
measures. Restricting a probability measure to an event of probability 0.2 leaves total
mass 0.2; dividing by 0.2 produces the conditional distribution [@bernardo1994bayesian]. Each operation
specifies whether it retains that mass or [normalizes it](../../semantics/programs.md#5-algebraic-laws-and-their-scope).

The compositional measure algebra behind the operator families of the
DSL (`Mixture`, `Restrict`, `Pushforward`, `PointMass`, ...). Each
combinator builds a measure from measures, and
[`normalize_at_boundary`](#quivers.continuous.measure.normalize_at_boundary)
renormalizes a sub-probability measure where it is sampled or scored.

::: quivers.continuous.measure
