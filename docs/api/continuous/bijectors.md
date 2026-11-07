# Bijectors

A bijector [changes coordinates with an inverse](../../guides/continuous-spaces.md#normalizing-flows). Exponentiation, for instance, converts
an unconstrained real into a positive value. Changing coordinates also changes [density
per unit volume](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample), so density calculations need the Jacobian correction [@rezende2015flows]. These interfaces
expose both the coordinate map and the correction needed by inference and transformed
distributions.

Invertible maps with log-Jacobian determinants, used to move between a
family's unconstrained parameters and its support, and by the
[`TransformReparam`](../effects/reparam.md) strategy.

::: quivers.continuous.bijectors
