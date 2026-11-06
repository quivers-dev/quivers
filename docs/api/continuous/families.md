# Continuous Families

Choose a family from the values the response can take and the variation you want to
model. A Normal models real-valued responses; positive and bounded supports need other
families. Conditional families derive distribution parameters from their input, so their
learned parameter map must be distinguished from a fixed call such as `Normal(loc=0,
scale=1)`.

See [choosing a distribution family](../../guides/continuous-families.md#family-registry) and
[constructing conditional families in Python](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families).

For distribution support and parameter conventions, consult the [PyTorch distribution reference](https://docs.pytorch.org/docs/stable/distributions.html).

Families of continuous-valued distributions.

::: quivers.continuous.families
