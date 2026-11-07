# Arrow Instances

These instances implement [arrow operations](../../guides/effects.md#typeclass-hierarchy) for Python functions, finite weighted
relations, and stochastic transitions [@hughes2000arrows]. Function composition passes one result to the
next function. Stochastic composition averages the next output distribution over the
possible intermediate values; each transition must be a normalized kernel.

The standard implementations cover finite weighted relations (`VRel`),
ordinary functions (`Function`), and stochastic transitions (`Stochastic`).

::: quivers.arrows.instances
