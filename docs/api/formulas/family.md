# `quivers.formulas.family`

A [response family](../../guides/analysis-data-and-formulas.md#family-registry) is the distribution of
the observations, and its [link function](https://en.wikipedia.org/wiki/Generalized_linear_model)
connects the linear predictor to a parameter of that distribution [@nelder1972glm]. In
[Bernoulli regression](../../tutorials/qvr/02-glms.md), the inverse logit maps an unconstrained
predictor to a probability. The module gives `Family`, `Link`, and `AuxParam`, and the `families`
and `links` registries.

::: quivers.formulas.family
