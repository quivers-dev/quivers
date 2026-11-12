# Transforms

These functions [reweight and normalize stochastic morphisms](../../guides/stochastic.md#conditioning-and-bayesian-updates).
`condition` multiplies each row by evidence weights and renormalizes it, which conditions the
output distribution on that evidence. `mix` forms a mixture of morphisms, `factor` reweights a
morphism, and `normalize` renormalizes its rows. Inferring the input instead needs a prior over
inputs weighted by the evidence likelihood of each, that is,
[Bayesian inversion](https://en.wikipedia.org/wiki/Bayes%27_theorem) [@cho2019disintegration].

::: quivers.stochastic.transforms
