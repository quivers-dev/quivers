# Bijectors

A bijector changes coordinates with an inverse. Exponentiation, for instance, converts
an unconstrained real into a positive value. Changing coordinates also changes density
per unit volume, so density calculations need the Jacobian correction. These interfaces
expose both the coordinate map and the correction needed by inference and transformed
distributions.

See [invertible transformations and density corrections](../../guides/continuous-spaces.md#normalizing-flows) and
[the distinction between mass, density, and a sampled value](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample).

For change of variables in probabilistic inference, see [@rezende2015flows].

Invertible maps with log-Jacobian determinants, used to move between a
family's unconstrained parameters and its support, and by the
[`TransformReparam`](../effects/reparam.md) strategy.

::: quivers.continuous.bijectors
