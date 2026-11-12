# Inline Distribution Builders

The DSL compiler emits these morphisms for an
[inline family call](../../guides/dsl-programs-and-lets.md#inline-distributions) such as
`sample x <- Normal(loc=mu, scale=1.0)`. A call whose arguments are all literals becomes a
[`FixedDistribution`](#quivers.continuous.inline.FixedDistribution), built by a `make_fixed_*`
factory. A call with at least one bound argument, such as `loc=beta * x`, becomes a
[`MixedInlineDistribution`](#quivers.continuous.inline.MixedInlineDistribution), which reads the
bound arguments from its stacked input and fills in the literals, so that
[a later draw can depend on an earlier one](../../reference/qvr/probabilistic-programs.md#step-reference).
Both can be constructed directly to build a program in Python.

::: quivers.continuous.inline
