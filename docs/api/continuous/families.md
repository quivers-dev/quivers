# Continuous Families

The module gives the families of continuous distributions
([family registry](../../guides/continuous-families.md#family-registry)), each chosen by the values
a response can take: a Normal for real values, and other families for positive or bounded
supports. A [conditional family](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families)
computes its parameters from its input through a learned parameter map, while a fixed call such as
`Normal(loc=0, scale=1)` has constant parameters.

::: quivers.continuous.families

## Asymmetric Laplace quantiles

`AsymmetricLaplace(loc, scale, quantile)` models a selected response quantile;
`ConditionalAsymmetricLaplace` learns all three parameters from its input. The
native distribution supplies differentiable sampling, density, CDF, inverse CDF,
mean, variance, and entropy. Its scale follows the check-loss convention used in
[brms quantile regression](https://paulbuerkner.com/brms/articles/brms_families.html#quantile-regression):
at quantile `0.5`, it equals `Laplace(loc, 2 * scale)`.

::: quivers.continuous._asymmetric_laplace

## Response-time distributions

`ExGaussian(loc, scale, rate)` is the sum of independent Normal and Exponential
variables, with response mean `loc + 1/rate`. `InverseGaussian(mu, shape)` uses
the response mean and positive shape directly. Both provide native densities,
CDFs, moments, broadcasting, and reparameterized sampling. Inverse Gaussian
samples use an exact vectorized transformation and implicit CDF derivatives;
its discrete sampler branch is not differentiated as though it were fixed.

::: quivers.continuous._response_time
