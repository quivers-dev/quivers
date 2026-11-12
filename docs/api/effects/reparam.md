# Reparameterization

Reparameterization rewrites a sample site in other variables with the same distribution. A
[location and scale transform](../../tutorials/qvr/03-hierarchical.md#non-centered-parameterization),
for instance, writes a Normal draw as a location plus a scale times a standard Normal draw
[@papaspiliopoulos2007parametrization]. Keeping the target density requires the transformed prior,
the observation scores, and any [Jacobian correction](../continuous/bijectors.md). Neural transport
learns a change of coordinates meant to simplify the posterior geometry before Markov chain Monte Carlo (MCMC)
[@hoffman2019neutra].

Reparameterization strategies for individual sample sites.

::: quivers.effects.reparam
