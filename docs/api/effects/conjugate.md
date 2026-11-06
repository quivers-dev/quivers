# Conjugate marginals

Conjugate updates combine a prior with a likelihood whose product belongs to a known
distribution family. A Beta prior and Bernoulli observations, for instance, update the
two shape parameters by adding success and failure counts. The supported combinations
below provide these analytic updates.

See [using conjugacy to integrate out a latent variable](collapse.md) and
[a Beta–Binomial model with an analytically marginalized rate](../../examples/beta-binomial-ab-test.md).

For conjugate analysis and marginalization, see [@bernardo1994bayesian].

The closed-form marginals of the standard conjugate pairs.

::: quivers.effects.conjugate
