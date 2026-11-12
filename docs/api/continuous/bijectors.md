# Bijectors

A bijector is an invertible differentiable map with its log Jacobian determinant
([normalizing flows](../../guides/continuous-spaces.md#normalizing-flows)). Exponentiation, for
instance, maps an unconstrained real to a positive value. A change of coordinates changes density
per unit volume, so a density computed through a bijector needs the
[Jacobian correction](https://mc-stan.org/docs/stan-users-guide/reparameterization.html)
[@rezende2015flows; @stanReparameterization]. Families use bijectors to map their unconstrained
parameters onto their supports, and the [`TransformReparam`](../effects/reparam.md) strategy uses
them to reparameterize a site.

::: quivers.continuous.bijectors
