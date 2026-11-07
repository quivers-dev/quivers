# Continuous Families

[Choose a family](../../guides/continuous-families.md#family-registry) from the [values the response can take](https://docs.pytorch.org/docs/stable/distributions.html) and the variation you want to
model. A Normal models real-valued responses; positive and bounded supports need other
families. [Conditional families](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families) derive distribution parameters from their input, so their
learned parameter map must be distinguished from a fixed call such as `Normal(loc=0,
scale=1)`.

Families of continuous-valued distributions.

::: quivers.continuous.families
