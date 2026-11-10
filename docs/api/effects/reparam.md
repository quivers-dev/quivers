# Reparameterization

Reparameterization changes the variables used to express a model. A [location-scale
transform](../../tutorials/qvr/03-hierarchical.md#non-centered-parameterization), for instance, writes a Normal variable as a location plus a scale times a
standard Normal draw. Preserving the target density requires accounting for the
transformed prior, observation scores, and any [Jacobian correction](../continuous/bijectors.md).

Neural transport learns a change of coordinates intended to simplify the
target geometry before running MCMC [@hoffman2019neutra].

Reparameterization strategies for individual sample sites.

::: quivers.effects.reparam
