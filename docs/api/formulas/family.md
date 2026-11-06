# `quivers.formulas.family`

The response family describes the distribution of observations; its link connects a
linear predictor to a distribution parameter. For Bernoulli regression, the inverse
logit maps an unconstrained predictor to a probability between zero and one. The
Bernoulli family models a response with two possible outcomes.

See [response families and links](../../guides/analysis-data-and-formulas.md#family-registry) and
[a Gaussian regression fit](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line).

For regression and partial pooling, see [@gelman2013bayesian].

Response families and link functions: `Family`, `Link`, `AuxParam`,
plus the `families` and `links` registries.

::: quivers.formulas.family
