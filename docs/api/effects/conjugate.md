# Conjugate marginals

Conjugate updates combine a prior with a likelihood whose product belongs to a known
distribution family [@bernardo1994bayesian]. A [Beta prior and Bernoulli observations](../../examples/beta-binomial-ab-test.md), for instance, update the
two shape parameters by adding success and failure counts. The supported combinations
below provide these [analytic updates](collapse.md).

The closed-form marginals of the standard conjugate pairs.

::: quivers.effects.conjugate
