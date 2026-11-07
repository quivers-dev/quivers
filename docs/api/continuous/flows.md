# Flows

A [normalizing flow](../../guides/continuous-spaces.md#normalizing-flows) transforms an easily sampled distribution through a sequence of
invertible maps. Sampling follows those maps. Density evaluation also uses their
Jacobian determinants to account for changes in volume, allowing the transformed
distribution to be scored during inference [@rezende2015flows].

Normalizing flows and their transformations.

::: quivers.continuous.flows
