# Reparameterization

Reparameterization changes the variables used to express a model. A location-scale
transform, for instance, writes a Normal variable as a location plus a scale times a
standard Normal draw. Preserving the target density requires accounting for the
transformed prior, observation scores, and any Jacobian correction.

See [a non-centered parameterization](../../tutorials/qvr/03-hierarchical.md#non-centered-parameterization) and
[coordinate transforms and density corrections](../continuous/bijectors.md).

For neural transport, see the author-published preprint [@hoffman2019neutra].

Reparameterization strategies for individual sample sites.

::: quivers.effects.reparam
