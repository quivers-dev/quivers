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

`zero_inflated_beta` uses logit mean and zero-probability links plus a log
precision link. It lowers to native `ZeroOneInflatedBeta` with `coi=0`, retaining
only the `phi` and `zi` auxiliaries. Its response mean includes the zero-atom
weight; see the [regression example](../../guides/analysis-data-and-formulas.md#zero-inflated-beta-regression).

`hurdle_gamma` uses positive-component mean/shape parameters;
`hurdle_lognormal` uses log-location/log-scale parameters. Both have a `hu`
auxiliary with a logit link and exact zero-atom semantics. Their response
expectations include the positive-component probability, as shown in the
[hurdle guide](../../guides/analysis-data-and-formulas.md#positive-hurdle-regressions).

::: quivers.formulas.family
