# Arrow Instances

These instances implement [arrow](https://en.wikipedia.org/wiki/Arrow_(computer_science))
operations ([type class hierarchy](../../guides/effects.md#typeclass-hierarchy)) for ordinary
functions (`Function`), finite weighted relations (`VRel`), and stochastic transitions
(`Stochastic`) [@hughes2000arrows]. Composing functions passes one result to the next function.
Composing stochastic transitions averages the next output distribution over the intermediate
values, so each transition must be a normalized kernel.

::: quivers.arrows.instances
