# Flows

A normalizing flow transforms an easily sampled distribution through a sequence of
invertible maps. Sampling follows those maps. Density evaluation also uses their
Jacobian determinants to account for changes in volume, allowing the transformed
distribution to be scored during inference.

See [composing invertible maps into a flow](../../guides/continuous-spaces.md#normalizing-flows) and
[the distinction between mass, density, and a sampled value](../../getting-started/concepts.md#5-a-distribution-is-more-than-one-sample).

For an introduction to normalizing flows, see [@rezende2015flows].

Normalizing flows and their transformations.

::: quivers.continuous.flows
