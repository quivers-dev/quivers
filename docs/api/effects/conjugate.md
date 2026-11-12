# Conjugate marginals

A [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) gives a posterior in its own
family [@bernardo1994bayesian]. A [Beta prior with Bernoulli observations](../../examples/beta-binomial-ab-test.md),
for instance, updates its two shape parameters by the counts of successes and failures. The module
gives the closed-form marginals of the standard conjugate pairs, which [`CollapseHandler`](collapse.md)
uses.

::: quivers.effects.conjugate
