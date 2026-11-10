# `quivers.formulas.family`

The [response family](../../guides/analysis-data-and-formulas.md#family-registry) describes the distribution of observations; its link connects a
linear predictor to a distribution parameter. For [Bernoulli regression](../../tutorials/qvr/02-glms.md), the inverse
logit maps an unconstrained predictor to a probability between zero and one. The
Bernoulli family models a response with two possible outcomes.

Response families and link functions: `Family`, `Link`, `AuxParam`,
plus the `families` and `links` registries.

::: quivers.formulas.family
