# `quivers.formulas.family`

A [response family](../../guides/analysis-data-and-formulas.md#family-registry) is the distribution of
the observations, and its [link function](https://en.wikipedia.org/wiki/Generalized_linear_model)
connects the linear predictor to a parameter of that distribution [@nelder1972glm]. In
[Bernoulli regression](../../tutorials/qvr/02-glms.md), the inverse logit maps an unconstrained
predictor to a probability. The module gives `Family`, `Link`, and `AuxParam`, and the `families`
and `links` registries. `Family.with_links(mu="probit")` selects a location link,
while `Family.with_links(sigma="softplus")` changes a distributional scale link;
[link selection](../../guides/analysis-data-and-formulas.md#distributional-regression-and-link-selection)
gives the supported names and parameterizations.

::: quivers.formulas.family
