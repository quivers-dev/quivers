# Measure Algebra

These combinators build measures from measures: `Mixture`, `Restrict`, `Pushforward`, `PointMass`,
`Independent`, and `Normalize`, which back the operator families of the DSL. Restricting a
probability measure to an event of probability 0.2 leaves total mass 0.2, and dividing by 0.2 gives
the conditional distribution [@bernardo1994bayesian]. Each combinator states whether it keeps that
mass or [normalizes it](../../semantics/programs.md#5-algebraic-laws-and-their-scope), and
[`normalize_at_boundary`](#quivers.continuous.measure.normalize_at_boundary) renormalizes a
subprobability measure where it is sampled or scored.

::: quivers.continuous.measure
