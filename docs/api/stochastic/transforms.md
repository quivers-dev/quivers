# Transforms

These transformations [reweight and normalize transition tables](../../guides/stochastic.md#conditioning-and-bayesian-updates). Multiplying a row by
evidence weights and normalizing conditions its output distribution on that evidence. To
infer the input state, also supply a prior over inputs and weight it by the evidence
likelihood for each input [@cho2019disintegration].

Transformations of stochastic morphisms.

::: quivers.stochastic.transforms
