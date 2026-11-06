# Transforms

These transformations reweight and normalize transition tables. Multiplying a row by
evidence weights and normalizing conditions its output distribution on that evidence. To
infer the input state, also supply a prior over inputs and weight it by the evidence
likelihood for each input.

See [reweighting and normalization](../../guides/stochastic.md#conditioning-and-bayesian-updates) and
[a Markov kernel calculation](../../tutorials/python/02-stochastic-relations.md#composition-of-markov-kernels).

For Bayesian inversion and the role of the prior, see [@cho2019disintegration].

Transformations of stochastic morphisms.

::: quivers.stochastic.transforms
