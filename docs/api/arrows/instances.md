# Arrow Instances

These instances implement arrow operations for Python functions, finite weighted
relations, and stochastic transitions. Function composition passes one result to the
next function. Stochastic composition averages the next output distribution over the
possible intermediate values; each transition must be a normalized kernel.

See [the arrow operations](../../guides/effects.md#typeclass-hierarchy) and
[the relationship to monadic sequencing](../monadic/bridges.md).

For the arrow interfaces and their motivation, see [@hughes2000arrows].

The standard implementations cover finite weighted relations (`VRel`),
ordinary functions (`Function`), and stochastic transitions (`Stochastic`).

For categorical trace axioms, see [@joyal1996traced].

::: quivers.arrows.instances
