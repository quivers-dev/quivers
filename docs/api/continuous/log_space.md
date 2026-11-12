# Log-Space Families

A [Beta](https://en.wikipedia.org/wiki/Beta_distribution) or
[Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) variable with a small shape parameter
keeps most of its mass within a few multiples of the smallest positive floating-point number, where
scoring it through a sigmoid or an exponential loses its tails. These distributions live on the
real line instead. [`LogitBeta`](#quivers.continuous.log_space.LogitBeta) is the distribution of the
[logit](https://en.wikipedia.org/wiki/Logit) of a Beta variable, and
[`LogGamma`](#quivers.continuous.log_space.LogGamma) that of the log of a Gamma variable, each with
its exact density and a sampler that stays in log space. The
[continuous families guide](../../guides/continuous-families.md#log-space-families-for-small-shapes)
says when to use them.

::: quivers.continuous.log_space
